import requests
import time
import threading
from concurrent.futures import ThreadPoolExecutor

# Configuration
API_URL = "http://localhost:5000"
TEXT_ENDPOINT = "/text"
NUM_REQUESTS = 10
CONCURRENT_REQUESTS = 5

def make_request(prompt, optimize=True, use_cache=True):
    """Make a single request to the API"""
    start_time = time.time()
    
    response = requests.post(
        f"{API_URL}{TEXT_ENDPOINT}",
        json={
            "prompt": prompt,
            "optimize": optimize,
            "use_cache": use_cache
        }
    )
    
    end_time = time.time()
    
    return {
        "status_code": response.status_code,
        "response_time": end_time - start_time,
        "cached": response.json().get("cached", False) if response.status_code == 200 else False
    }

def benchmark_sequential():
    """Benchmark sequential requests"""
    print("Running sequential benchmark...")
    results = []
    
    for i in range(NUM_REQUESTS):
        prompt = f"Explain concept {i} in computer science"
        result = make_request(prompt)
        results.append(result)
        print(f"Request {i+1}: {result['response_time']:.2f}s {'(cached)' if result['cached'] else ''}")
    
    return results

def benchmark_concurrent():
    """Benchmark concurrent requests"""
    print("\nRunning concurrent benchmark...")
    results = []
    
    def worker(i):
        prompt = f"Explain concept {i} in computer science"
        result = make_request(prompt)
        print(f"Request {i+1}: {result['response_time']:.2f}s {'(cached)' if result['cached'] else ''}")
        return result
    
    with ThreadPoolExecutor(max_workers=CONCURRENT_REQUESTS) as executor:
        futures = [executor.submit(worker, i) for i in range(NUM_REQUESTS)]
        results = [future.result() for future in futures]
    
    return results

def benchmark_with_caching():
    """Benchmark with caching enabled"""
    print("\nRunning benchmark with caching...")
    
    # First request (not cached)
    result1 = make_request("Explain quantum computing")
    print(f"First request: {result1['response_time']:.2f}s {'(cached)' if result1['cached'] else ''}")
    
    # Second request (should be cached)
    result2 = make_request("Explain quantum computing")
    print(f"Second request: {result2['response_time']:.2f}s {'(cached)' if result2['cached'] else ''}")
    
    return [result1, result2]

def benchmark_without_optimizations():
    """Benchmark without optimizations"""
    print("\nRunning benchmark without optimizations...")
    results = []
    
    for i in range(3):  # Fewer requests to save time
        prompt = f"Explain concept {i} in computer science"
        result = make_request(prompt, optimize=False, use_cache=False)
        results.append(result)
        print(f"Request {i+1}: {result['response_time']:.2f}s {'(cached)' if result['cached'] else ''}")
    
    return results

def calculate_stats(results):
    """Calculate statistics from benchmark results"""
    if not results:
        return {}
    
    response_times = [r['response_time'] for r in results if r['status_code'] == 200]
    cached_count = sum(1 for r in results if r.get('cached', False))
    
    if not response_times:
        return {}
    
    return {
        "total_requests": len(results),
        "successful_requests": len(response_times),
        "average_response_time": sum(response_times) / len(response_times),
        "min_response_time": min(response_times),
        "max_response_time": max(response_times),
        "cached_requests": cached_count
    }

if __name__ == "__main__":
    print("Benchmarking Optimized Multimodal API")
    print("=" * 40)
    
    # Run benchmarks
    seq_results = benchmark_sequential()
    conc_results = benchmark_concurrent()
    cache_results = benchmark_with_caching()
    no_opt_results = benchmark_without_optimizations()
    
    # Calculate and display statistics
    print("\n" + "=" * 40)
    print("BENCHMARK RESULTS")
    print("=" * 40)
    
    seq_stats = calculate_stats(seq_results)
    print("\nSequential Requests:")
    print(f"  Average Response Time: {seq_stats.get('average_response_time', 0):.2f}s")
    print(f"  Cached Requests: {seq_stats.get('cached_requests', 0)}/{seq_stats.get('total_requests', 0)}")
    
    conc_stats = calculate_stats(conc_results)
    print("\nConcurrent Requests:")
    print(f"  Average Response Time: {conc_stats.get('average_response_time', 0):.2f}s")
    print(f"  Cached Requests: {conc_stats.get('cached_requests', 0)}/{conc_stats.get('total_requests', 0)}")
    
    cache_stats = calculate_stats(cache_results)
    print("\nCaching Benefits:")
    print(f"  First Request: {cache_results[0]['response_time']:.2f}s")
    print(f"  Cached Request: {cache_results[1]['response_time']:.2f}s")
    if cache_results[0]['response_time'] > 0:
        improvement = (1 - cache_results[1]['response_time'] / cache_results[0]['response_time']) * 100
        print(f"  Improvement: {improvement:.1f}%")
    
    no_opt_stats = calculate_stats(no_opt_results)
    print("\nWithout Optimizations:")
    print(f"  Average Response Time: {no_opt_stats.get('average_response_time', 0):.2f}s")
    
    print("\nBenchmark completed!")