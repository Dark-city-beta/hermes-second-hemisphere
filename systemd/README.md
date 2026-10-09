# Production Systemd Configuration for Auxiliary Brain

Running the auxiliary LLM on CPU or a dedicated lightweight compute slot requires careful tuning to avoid performance cliffs.

### Key Tuning Rules

1. **Flash Attention (`-fa on`)**:
   - Reduces memory usage and speeds up prompt processing for longer contexts (e.g. 3,000–8,000 tokens) even on CPU.

2. **Thread Allocation (`-t N`)**:
   - Set `N` to the physical core count of your CPU (e.g. 4 for a 4-core / 8-thread Xeon/i7).
   - Exceeding physical core count often causes hyperthread cache thrashing during matrix multiplications.

3. **Cgroup Memory Throttle Trap (`MemoryHigh`)**:
   - **Warning**: Setting `MemoryHigh` too close to the process baseline (e.g. 2.5 GB for a 1.5B model with 16k KV cache) causes Linux kernel memory reclaim loops.
   - In our production tests, memory high pressure dropped token generation from **15.0 t/s** down to **0.26 t/s** (a 50x slowdown!).
   - Ensure at least 1.5–2 GB headroom above the model size:
     ```ini
     MemoryHigh=4500M
     MemoryMax=6G
     MemorySwapMax=2G
     ```

4. **Time Sync Order (`After=time-sync.target`)**:
   - Always ensure network services and VPN tunnels start after NTP time synchronization to prevent TLS certificate validation failures on boot.
