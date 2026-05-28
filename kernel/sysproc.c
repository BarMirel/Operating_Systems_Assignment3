#include "types.h"
#include "riscv.h"
#include "defs.h"
#include "param.h"
#include "memlayout.h"
#include "spinlock.h"
#include "proc.h"

uint64
sys_exit(void)
{
  int n;
  argint(0, &n);
  exit(n);
  return 0;  // not reached
}

uint64
sys_getpid(void)
{
  return myproc()->pid;
}

uint64
sys_fork(void)
{
  return fork();
}

uint64
sys_wait(void)
{
  uint64 p;
  argaddr(0, &p);
  return wait(p);
}

uint64
sys_sbrk(void)
{
  uint64 addr;
  int n;

  argint(0, &n);
  addr = myproc()->sz;
  if(growproc(n) < 0)
    return -1;
  return addr;
}

uint64
sys_sleep(void)
{
  int n;
  uint ticks0;

  argint(0, &n);
  acquire(&tickslock);
  ticks0 = ticks;
  while(ticks - ticks0 < n){
    if(killed(myproc())){
      release(&tickslock);
      return -1;
    }
    sleep(&ticks, &tickslock);
  }
  release(&tickslock);
  return 0;
}

uint64
sys_kill(void)
{
  int pid;

  argint(0, &pid);
  return kill(pid);
}

// return how many clock tick interrupts have occurred
// since start.
uint64
sys_uptime(void)
{
  uint xticks;

  acquire(&tickslock);
  xticks = ticks;
  release(&tickslock);
  return xticks;
}

// sys_flip_display: zero-copy page flip.
//
// Argument 0: user virtual address of a page-aligned buffer
// (GPU_FB_PAGES * PGSIZE bytes) already mapped in this process.
// Re-points the GPU's backing list to the user buffer's physical pages.
// Returns 0 on success, -1 on error.
uint64
sys_flip_display(void)
{
  uint64 va;
  struct proc *p = myproc();

  argaddr(0, &va);

  // Buffer must be page-aligned.
  if(va % PGSIZE != 0)
    return -1;

  // All GPU_FB_PAGES pages must be mapped with user permission.
  for(int i = 0; i < GPU_FB_PAGES; i++){
    if(walkaddr(p->pagetable, va + (uint64)i * PGSIZE) == 0)
      return -1;
  }

  virtio_gpu_flip(p->pagetable, va);
  p->display_flip_active = 1;
  p->display_flip_va = va;
  return 0;
}

// sys_map_display: map the GPU's kernel framebuffer pages (fb[]) directly
// into the calling process's address space with PTE_U|PTE_R|PTE_W.
//
// Argument 0: desired user virtual address (must be page-aligned).
//   Pass 0 to let the kernel auto-select a VA just above p->sz.
// Returns the mapped VA on success, (uint64)-1 on failure.
// The framebuffer pages are kernel-owned; they must NOT be freed on unmap.
uint64
sys_map_display(void)
{
  uint64 addr;
  struct proc *p = myproc();

  argaddr(0, &addr);

  uint64 va;
  if(addr == 0){
    // Kernel chooses: first page-aligned address above the heap.
    va = PGROUNDUP(p->sz);
  } else {
    // User-supplied address: must be page-aligned.
    if(addr % PGSIZE != 0)
      return -1;
    va = addr;
    // Check that none of the GPU_FB_PAGES pages in [va, va+size) are mapped.
    for(int i = 0; i < GPU_FB_PAGES; i++){
      pte_t *pte = walk(p->pagetable, va + (uint64)i * PGSIZE, 0);
      if(pte != 0 && (*pte & PTE_V))
        return -1;
    }
  }

  // Map each framebuffer page individually (fb[] pages are not contiguous).
  for(int i = 0; i < GPU_FB_PAGES; i++){
    uint64 pa = virtio_gpu_fb_pa(i);
    if(mappages(p->pagetable, va + (uint64)i * PGSIZE, PGSIZE,
                pa, PTE_R | PTE_W | PTE_U) != 0){
      // Unmap the pages already installed (without freeing — they're kernel pages).
      if(i > 0)
        uvmunmap(p->pagetable, va, i, 0);
      return -1;
    }
  }

  p->display_map_va = va;
  return va;
}
