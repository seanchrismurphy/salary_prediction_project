import time
import random
import json
import os
import asyncio
from datetime import datetime
from playwright.async_api import async_playwright


class RedirectResolver:
    async def resolve_redirect(self, redirect_url, max_wait=30):
        if "/details/" in redirect_url:
            return redirect_url

        async with async_playwright() as p:
            browser = await p.firefox.launch(headless=True)

            context = await browser.new_context(
                user_agent="Mozilla/5.0 (X11; Linux x86_64; rv:121.0) Gecko/20100101 Firefox/121.0",
                viewport={"width": 1366, "height": 768},
                locale="en-AU",
                timezone_id="Australia/Melbourne",
            )

            page = await context.new_page()

            await page.route("**/*", lambda route:
                route.abort() if any(x in route.request.url for x in [
                    "datadome", "fingerprint", "captcha", "bot-detect"
                ]) else route.continue_()
            )

            try:
                await page.goto(redirect_url, timeout=30000, wait_until="commit")

                wait_time = 0
                last_url = redirect_url
                stable_count = 0

                while wait_time < max_wait:
                    await page.wait_for_timeout(2000)
                    wait_time += 2

                    current_url = page.url

                    if current_url == last_url:
                        stable_count += 1
                        if stable_count >= 2 and current_url != redirect_url:
                            return current_url
                    else:
                        stable_count = 0
                        last_url = current_url

                if page.url != redirect_url:
                    return page.url

                return None

            except Exception as e:
                print(f"Error resolving {redirect_url[:70]}: {e}")
                return None
            finally:
                await browser.close()


def save_progress(successful_results, failed_urls, current_index, total_urls):
    try:
        with open("data/progress_tracking/redirect_progress.json", "w") as f:
            json.dump({
                "last_processed": current_index,
                "successful_results": successful_results,
                "failed_urls": failed_urls,
                "timestamp": datetime.now().isoformat(),
                "total_urls": total_urls,
            }, f, indent=2)

        print(f"Progress saved: {len(successful_results)} successful, {len(failed_urls)} failed")

    except Exception as e:
        print(f"Error saving progress: {e}")


async def process_redirect_urls(redirect_urls, save_interval=50):
    resolver = RedirectResolver()
    total_urls = len(redirect_urls)
    successful_results = []
    failed_urls = []
    current_batch_start = 0

    progress_file = "data/progress_tracking/redirect_progress.json"
    if os.path.exists(progress_file):
        try:
            with open(progress_file, "r") as f:
                progress_data = json.load(f)
                current_batch_start = progress_data.get("last_processed", 0)
                successful_results = progress_data.get("successful_results", [])
                failed_urls = progress_data.get("failed_urls", [])
            print(f"Resuming from URL #{current_batch_start + 1}")
            print(f"Previous: {len(successful_results)} successful, {len(failed_urls)} failed")
        except Exception as e:
            print(f"Error reading progress file: {e}. Starting fresh.")
            current_batch_start = 0

    print(f"\nProcessing {total_urls} URLs, saving every {save_interval}")
    start_time = datetime.now()

    try:
        for i in range(current_batch_start, total_urls):
            url_number = i + 1
            current_url = redirect_urls[i]

            print(f"\n[{url_number}/{total_urls}] ({url_number/total_urls*100:.1f}%)")

            success = False
            max_retries = 2

            for attempt in range(max_retries):
                if attempt > 0:
                    retry_delay = random.uniform(120, 300)
                    print(f"Retry {attempt + 1}/{max_retries} in {retry_delay/60:.1f} min...")
                    await asyncio.sleep(retry_delay)

                try:
                    url_start = time.time()
                    final_url = await resolver.resolve_redirect(current_url, max_wait=30)
                    elapsed = time.time() - url_start

                    if final_url and (final_url != current_url or "/details/" in current_url):
                        successful_results.append({
                            "index": i,
                            "original_url": current_url,
                            "final_url": final_url,
                            "processing_time": elapsed,
                            "attempts": attempt + 1,
                            "timestamp": datetime.now().isoformat(),
                        })
                        print(f"Success in {elapsed:.1f}s: {final_url[:70]}")
                        success = True
                        break
                    else:
                        print(f"No redirect on attempt {attempt + 1}")

                except Exception as e:
                    print(f"Error on attempt {attempt + 1}: {e}")

            if not success:
                failed_urls.append({
                    "index": i,
                    "url": current_url,
                    "failed_at": datetime.now().isoformat(),
                    "attempts": max_retries,
                })
                print(f"Failed after {max_retries} attempts")

            if url_number % save_interval == 0 or url_number == total_urls:
                save_progress(successful_results, failed_urls, url_number, total_urls)

                elapsed_total = datetime.now() - start_time
                avg_per_url = elapsed_total.total_seconds() / url_number
                remaining_hrs = (total_urls - url_number) * avg_per_url / 3600

                print(f"Checkpoint: {len(successful_results)} ok, {len(failed_urls)} failed")
                print(f"Elapsed: {elapsed_total}, est remaining: {remaining_hrs:.1f}h")

                if url_number < total_urls:
                    if "/details/" in current_url:
                        break_time = 0
                    else:
                        break_time = random.uniform(120, 180)
                    print(f"Break: {break_time/60:.1f} min...")
                    await asyncio.sleep(break_time)
            else:
                if url_number < total_urls:
                    if "/details/" in current_url:
                        delay = 0
                    else: delay = random.uniform(15, 30)
                    print(f"Next in {delay:.0f}s...")
                    await asyncio.sleep(delay)

    except KeyboardInterrupt:
        print(f"\nInterrupted at URL {url_number}")
        save_progress(successful_results, failed_urls, url_number, total_urls)
        return successful_results, failed_urls

    except Exception as e:
        print(f"\nUnexpected error: {e}")
        save_progress(successful_results, failed_urls, url_number, total_urls)
        return successful_results, failed_urls

    total_elapsed = datetime.now() - start_time
    print(f"\nDone. {len(successful_results)}/{total_urls} successful in {total_elapsed}")
    save_progress(successful_results, failed_urls, total_urls, total_urls)

    return successful_results, failed_urls


if __name__ == "__main__":
    urls_file = "data/processed/redirect_urls.json"
    if not os.path.exists(urls_file):
        print(f"URL file not found: {urls_file}")
        print("Save your redirect_urls list as JSON first:")
        print("  import json")
        print("  with open('data/processed/redirect_urls.json', 'w') as f:")
        print("      json.dump(redirect_urls, f)")
        exit(1)

    with open(urls_file, "r") as f:
        redirect_urls = json.load(f)

    print(f"Loaded {len(redirect_urls)} URLs from {urls_file}")
    asyncio.run(process_redirect_urls(redirect_urls))
