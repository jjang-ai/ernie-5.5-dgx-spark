# Optional 64 KiB Linux Kernel + `nvidia-uvm` 4 KiB Sub-Page GPU Page-Table Slab

**Optional OS/kernel optimization for NVIDIA DGX Spark (`GB10`, 128 GiB unified RAM).**  
Reclaims **`~1.8 GiB` of net usable unified RAM per DGX Spark** compared to a stock `4 KiB` Linux kernel without changing vLLM, model weights, arithmetic, or serving configs.

---

## Why This Works (First-Principles Unified-Memory Accounting)

On a `128 GiB` ARM64 system (`33,554,432` `4 KiB` frames):
1. **Why a `64 KiB` kernel saves host RAM:** Every physical page needs a 64-byte `struct page` in the kernel `vmemmap` array. Switching from `4 KiB` to `64 KiB` host pages cuts the `vmemmap` footprint by `16×`: **`2.00 GiB → 0.125 GiB`** (**`+1.875 GiB` reclaimed per Spark**).
2. **Why a stock `64 KiB` kernel + stock `nvidia-uvm` loses that gain:** In `nvidia-uvm/uvm_mmu.c`, `phys_mem_allocate_sysmem()` allocates an entire host CPU page (`alloc_pages(flags, get_order(size))`) for every GPU page table or directory. On a `64 KiB` kernel, even a `256 B` or `4 KiB` GPU leaf page table consumes a full `64 KiB` backing page — wasting **`~3+ GiB`** of unified RAM when large LLM weights and KV pools are mapped!
3. **The fix (16 × `4 KiB` slots per `64 KiB` page):** [`patches/nvidia-580-uvm-subpage-pt.patch`](patches/nvidia-580-uvm-subpage-pt.patch) adds a per-`uvm_page_tree_t` `4 KiB` sub-page slab allocator (compiled strictly behind `#if PAGE_SIZE > 4096`) to `nvidia-uvm`. Up to `16` non-root GPU page tables (`<= 4 KiB`) share a single DMA-mapped `64 KiB` host page, eliminating the `~3 GiB` fragmentation tax and delivering the **`~1.8 GiB/Spark`** net unified-RAM gain over `4 KiB` Linux.

---

## Architecture, Scope & Safety Invariants

- **Strict Scope (`#if PAGE_SIZE > 4096`):** Compiles to stock behavior on `4 KiB` kernels. Runtime kill-switch `nvidia_uvm.uvm_pt_subpage_alloc=0` disables packing without rebuilding.
- **Root PDB Excluded (`depth == 0`):** `nvUvmInterfaceSetPageDirectory()` registers the page-tree root via `page_to_phys(tree->root->phys_alloc.handle.page)` without a sub-page offset. Root directories (`depth == 0`) always retain dedicated full-page allocations.
- **Size & Alignment Guards:** Only sysmem allocations with `0 < size <= 4096` use sub-page slots. Every backing page is verified `PAGE_ALIGNED(dma_addr)` before entering the slab pool (otherwise unmapped and cleanly fallen back to stock allocation), guaranteeing `offset_in_page(phys_alloc->addr.address) == slot * 4096` for `uvm_mmu_page_table_cpu_map()`.
- **Locking & Lifetime Rules:**
  - `allocate_directory()` can run *without* `tree->lock`, whereas `phys_mem_deallocate_sysmem()` runs *with* `tree->lock`. A dedicated per-tree `spinlock_t` protects only the O(1) `partial_slabs` / `full_slabs` lists and bitmask updates; all sleeping `alloc_pages`, DMA map/unmap, and `__free_pages` calls run strictly outside the spinlock.
  - `phys_mem_deallocate_sysmem()` always finishes `uvm_tracker_wait(&tree->tracker)` before returning a slot to its slab, ensuring all GPU CE writes and TLB invalidations referencing that slot have retired.
  - Reused `4 KiB` slots are explicitly zeroed (`kmap` + `memset` + `wmb`) before returning to the caller.
  - Empty slabs (`used_mask == 0`) are immediately unlinked, DMA-unmapped, and returned to the kernel page allocator. `uvm_page_tree_deinit()` asserts both slab lists are empty.
- **Per-Process CGroup Isolation:** Slabs are owned strictly per `uvm_page_tree_t` (never shared across processes or cgroups) and preserve stock `uvm_memcg_context_start/end` accounting.

---

## Quick Start (DGX OS 7.6.0 / Ubuntu 24.04 + NVIDIA `580.x` Open Driver)

### 1. Check Current Node State (Read-Only)
```bash
bash tools/uvm_64k_slab/spark_uvm_64k_slab.sh status
```

### 2. Install the Signed `64 KiB` NVIDIA Kernel & Open Driver Modules
```bash
sudo apt-get update
sudo apt-get install -y \
  dkms \
  linux-nvidia-64k-7.0 \
  linux-modules-nvidia-580-open-7.0.0-1025-nvidia-64k
```
*(Match the `linux-modules-nvidia-580-open-<ver>-nvidia-64k` version to the installed `7.0.0-*-nvidia-64k` kernel.)*

### 3. Dry-Run Build (Optional, Non-Destructive)
```bash
bash tools/uvm_64k_slab/spark_uvm_64k_slab.sh build --kernel 7.0.0-1025-nvidia-64k
```

### 4. Install Persistently via DKMS (with Secure Boot MOK Signing)
If Secure Boot is enabled (`mokutil --sb-state`), ensure you have an enrolled MOK keypair (for example `/var/lib/shim-signed/mok/MOK.priv` and `MOK.der` via `mokutil --import`), then install the override module:
```bash
sudo bash tools/uvm_64k_slab/spark_uvm_64k_slab.sh install-dkms \
  --kernel 7.0.0-1025-nvidia-64k \
  --sign-key /var/lib/shim-signed/mok/MOK.priv \
  --sign-cert /var/lib/shim-signed/mok/MOK.der
```
Select the `7.0.0-*-nvidia-64k` kernel in GRUB (`grub-set-default` or boot menu) and reboot.

### 5. Verify Live Savings
After booting into the `64 KiB` kernel and loading vLLM:
```bash
bash tools/uvm_64k_slab/spark_uvm_64k_slab.sh status
cat /sys/module/nvidia_uvm/parameters/uvm_pt_subpage_stats
```
Example output fields: `active_slots`, `active_slabs`, `peak_slots`, `peak_slabs`, and `saved_kib` (backing-page memory avoided vs stock `64 KiB` `nvidia-uvm`).

### 6. Clean Revert to Stock `nvidia-uvm`
```bash
sudo bash tools/uvm_64k_slab/spark_uvm_64k_slab.sh uninstall --kernel 7.0.0-1025-nvidia-64k
```
Reboot (or reload `nvidia_uvm`) to return to the stock signed Ubuntu module, or pick the `4 KiB` kernel in GRUB at any time.
