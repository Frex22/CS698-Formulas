Here is the summary and cheat sheet based on the Siebert and Traff parallel merge algorithm described in the text.

### Conceptual Summary
Parallel merging works by dividing the **output** array ($C$) into independent chunks for each thread. To fill a chunk starting at index $k$, the **Co-Rank Function** performs a binary search to find a specific split point $i$ in array $A$ and $j$ in array $B$ (where $i+j=k$). These split points ensure that all elements preceding $A[i]$ and $B[j]$ are smaller than the elements following them, allowing threads to merge their specific sub-arrays simultaneously without communicating with one another.

---

### Parallel Merge Co-Rank Cheat Sheet

#### 1. The Fundamental Variables
*   **$k$ (Rank):** The index in the output array $C$ we are trying to fill.
*   **$i$ (Co-rank of $k$ in $A$):** The number of elements taken from array $A$ to fill $C$ up to index $k$.
*   **$j$ (Co-rank of $k$ in $B$):** The number of elements taken from array $B$.
*   **The Golden Rule:** $i + j = k$

#### 2. Initialization Formulas
Before starting the search loop for a specific rank $k$:

| Variable | Formula | Logic |
| :--- | :--- | :--- |
| **Initial $i$** | $\min(k, m)$ | Start assuming we take as much from $A$ as possible (capped by $A$'s size $m$). |
| **Initial $j$** | $k - i$ | The rest must come from $B$. |
| **$i\_low$** | $\max(0, k - n)$ | If $k$ exceeds $B$'s size ($n$), we *must* take at least $k-n$ elements from $A$. |
| **$j\_low$** | $\max(0, k - m)$ | If $k$ exceeds $A$'s size ($m$), we *must* take at least $k-m$ elements from $B$. |

#### 3. The Co-Rank Search Logic (Iterative Loop)
Execute this loop to find the correct split. The goal is to satisfy stability ($A$ comes before $B$ on ties).

**Calculate Delta:**
*   $\text{delta}_i = \lfloor (i - i\_low + 1) / 2 \rfloor$
*   $\text{delta}_j = \lfloor (j - j\_low + 1) / 2 \rfloor$

**Check 1: Is $i$ too high? (Value in A is bigger than Value in B)**
*   **Condition:** $i > 0$ AND $j < n$ AND $A[i-1] > B[j]$
*   **Action:** Reduce $i$, Increase $j$.
    *   $j\_low = j$
    *   $j = j + \text{delta}_i$
    *   $i = i - \text{delta}_i$

**Check 2: Is $j$ too high? (Value in B is $\ge$ Value in A)**
*   **Condition:** $j > 0$ AND $i < m$ AND $B[j-1] \ge A[i]$
    *   *Note: Using $\ge$ ensures stability (if values are equal, we prefer $A$, so we reduce $j$ to take more from $A$).*
*   **Action:** Reduce $j$, Increase $i$.
    *   $i\_low = i$
    *   $i = i + \text{delta}_j$
    *   $j = j - \text{delta}_j$

**Success:**
*   **Condition:** Neither of the above checks passes.
*   **Result:** You have found the correct $i$ and $j$.

#### 4. Thread Work Assignment
If you have $P$ threads and want to merge $A$ (size $m$) and $B$ (size $n$):

1.  **Calculate Rank Scope:**
    *   Total elements = $m + n$.
    *   Elements per thread = $(m + n) / P$.
    *   Thread $t$'s start rank ($k_{start}$) = $t \times \text{Elements per thread}$.
    *   Thread $t$'s end rank ($k_{end}$) = $(t+1) \times \text{Elements per thread}$ (or total size for last thread).

2.  **Calculate Input Ranges (Using Co-Rank):**
    *   Call `co_rank(k_start)` $\rightarrow$ get $i_{start}, j_{start}$.
    *   Call `co_rank(k_end)` $\rightarrow$ get $i_{end}, j_{end}$.

3.  **Perform Sequential Merge:**
    *   Thread $t$ performs a standard sequential merge using:
        *   **Input A:** $A[i_{start} \dots i_{end}-1]$
        *   **Input B:** $B[j_{start} \dots j_{end}-1]$
        *   **Output C:** $C[k_{start} \dots k_{end}-1]$
     
Assume that we need to merge two lists A=(1, 7, 8, 9, 10) and B=(7, 10, 10, 12). What are the co-rank values for C[8]?
2. Complete the calculation of co-rank functions for thread 2 in Fig. 12.6.
3. For the for-loops that load A and B tiles in Fig. 12.12, add a call to the co-rank function so that we can load only the A and B elements that will be consumed in the current generation of the while-loop.

Here are the answers to your specific questions based on the parallel merge logic.

### 1. Co-rank values for C[8]

**Given:**
*   $A = \{1, 7, 8, 9, 10\}$ (Size $m=5$)
*   $B = \{7, 10, 10, 12\}$ (Size $n=4$)
*   **Target Rank ($k$):** 8

**Goal:** Find $i$ and $j$ such that $i+j=8$.

**Calculation:**
1.  **Initialization:**
    *   $i = \min(8, 5) = 5$
    *   $j = 8 - 5 = 3$
    *   $i\_low = \max(0, 8-4) = 4$
    *   $j\_low = \max(0, 8-5) = 3$
    *   *Analysis:* The window is very small. $i$ can only be 4 or 5.

2.  **Iteration 1:**
    *   **Check 1 (Is $i$ too high?):** Is $A[i-1] > B[j]$?
        *   $A[4] = 10$
        *   $B[3] = 12$
        *   $10 > 12$? **No.**
    *   **Check 2 (Is $j$ too high?):** Is $B[j-1] \ge A[i]$?
        *   $B[2] = 10$
        *   $A[5]$ is Out of Bounds (conceptually $\infty$).
        *   Is $10 \ge \infty$? **No.**

3.  **Result:** The loop terminates immediately.

**Answer:**
The co-rank values are **$i=5$** and **$j=3$**.
*(Meaning: To generate the first 8 elements of C, we consume all 5 elements of A and the first 3 elements of B.)*

---

### 2. Complete the co-rank calculation for Thread 2 (Fig 12.6)

**Context:**
In the text description for Figure 12.6, there are 3 threads merging a total of 9 elements.
*   Thread 0: $C[0]-C[2]$
*   Thread 1: $C[3]-C[5]$
*   **Thread 2: $C[6]-C[8]$**

Thread 2 needs to calculate the start point, which corresponds to **Rank $k=6$**.

**Arrays:**
*   $A = \{1, 7, 8, 9, 10\}$ ($m=5$)
*   $B = \{7, 10, 10, 12\}$ ($n=4$)

**Step-by-Step for $k=6$:**

1.  **Initialization:**
    *   $i = \min(6, 5) = \mathbf{5}$
    *   $j = 6 - 5 = \mathbf{1}$
    *   $i\_low = \max(0, 6-4) = \mathbf{2}$
    *   $j\_low = \max(0, 6-5) = \mathbf{1}$

2.  **Iteration 1 (Current $i=5, j=1$):**
    *   **Check 1 ($i$ too high?):** Is $A[i-1] > B[j]$?
        *   $A[4] = 10$
        *   $B[1] = 10$
        *   Is $10 > 10$? **No.** (Strict inequality fails).
    *   **Check 2 ($j$ too high?):** Is $B[j-1] \ge A[i]$?
        *   $B[0] = 7$
        *   $A[5] = \text{Out of Bounds}$.
        *   Is $7 \ge \infty$? **No.**

**Answer for Thread 2 Start:**
The search terminates immediately. The start co-ranks for Thread 2 are **$i=5$** and **$j=1$**.

---

### 3. Modifying Loop for Loading Tiles (Fig 12.12 context)

Although Figure 12.12 is not explicitly shown, the standard approach for a Tiled Parallel Merge involves loading a "tile" of A and B into Shared Memory. To do this, we must identify exactly which elements from A and B are required to produce the current output tile of C.

We use `co_rank` to find the start and end of the input ranges.

**Code Modification Example:**

```c
// Assume we are in a kernel. 
// tile_size is the number of C elements this block produces.
// c_start is the global index in C this block starts at.
int c_start = blockIdx.x * tile_size;
int c_end   = min(c_start + tile_size, m + n);

// 1. CALL CO-RANK TO FIND INPUT BOUNDARIES
// Find start indices (i_start, j_start) for this tile
int i_start = co_rank(c_start, A, m, B, n);
int j_start = c_start - i_start;

// Find end indices (i_end, j_end) for this tile
int i_end   = co_rank(c_end, A, m, B, n);
int j_end   = c_end - i_end;

// 2. LOAD A TILES
// We only load elements from A[i_start] to A[i_end]
// t is the thread index (threadIdx.x)
for (int t = threadIdx.x; t < (i_end - i_start); t += blockDim.x) {
    A_shared[t] = A[i_start + t];
}

// 3. LOAD B TILES
// We only load elements from B[j_start] to B[j_end]
for (int t = threadIdx.x; t < (j_end - j_start); t += blockDim.x) {
    B_shared[t] = B[j_start + t];
}

__syncthreads();
// Proceed with sequential merge using A_shared and B_shared...
```

Consider a parallel merge of two arrays of size 1,030,400 and 608,000. Assume that each thread merges eight elements and that a thread block size of 1024 is used.
a. In the basic merge kernel in Fig. 12.9, how many threads perform a binary search on the data in the global memory?
b. In the tiled merge kernel in Figs. 12.11–12.13, how many threads perform a binary search on the data in the global memory?
c. In the tiled merge kernel in Figs. 12.11–12.13, how many threads perform a binary search on the data in the shared memory?

Here are the step-by-step calculations and answers.

Preliminary Calculations:
First, let's calculate the total problem size and the grid dimensions required.

Total Elements (
𝑁
N
): Size A + Size B = 
1
,
030
,
400
+
608
,
000
=
1
,
638
,
400
1,030,400+608,000=1,638,400
 elements.

Elements per Thread: 8.

Total Threads Required: 
1
,
638
,
400
/
8
=
204
,
800
1,638,400/8=204,800
 threads.

Threads per Block: 1024.

Total Blocks Required: 
204
,
800
/
1024
=
200
204,800/1024=200
 blocks.

a. Basic Merge Kernel (Fig. 12.9)

Question: How many threads perform a binary search on the data in the global memory?

Analysis:
In the basic parallel merge approach (without tiling/shared memory), every single thread is responsible for generating a specific section of the output array 
𝐶
C
. To do this, every thread must independently determine where its specific section starts in input arrays 
𝐴
A
 and 
𝐵
B
. This requires every thread to call the co_rank function on the global arrays.

Calculation:
Total Threads = 204,800.

Answer:
204,800 threads perform a binary search on global memory.

b. Tiled Merge Kernel (Global Memory Search)

Question: How many threads perform a binary search on the data in the global memory?

Analysis:
In the tiled merge kernel, the goal is to reduce global memory traffic. Instead of every thread searching global memory, the Thread Block as a whole identifies a large "tile" of data (8192 elements in this case) to load into shared memory.

To define this tile, the block needs to find its Start Rank and End Rank in global memory.

Typically, 2 threads per block perform these searches (e.g., Thread 0 finds the start coordinates, and another thread—often the last one—finds the end coordinates).

The rest of the threads wait for these coordinates to be found so they can help load the data.

Calculation:

Total Blocks = 200.

Searches per Block = 2 (Start and End).

Total Threads = 
200
 blocks
×
2
 threads/block
=
400
200 blocks×2 threads/block=400
.

Answer:
400 threads perform a binary search on global memory.

c. Tiled Merge Kernel (Shared Memory Search)

Question: How many threads perform a binary search on the data in the shared memory?

Analysis:
Once the data tile is loaded into Shared Memory, the problem becomes a local merge sort within that block.

Each thread in the block is still responsible for merging 8 specific elements.

However, the thread now looks for its co-rank indices within the Shared Memory arrays (
𝐴
_
𝑠
ℎ
𝑎
𝑟
𝑒
𝑑
A_shared
 and 
𝐵
_
𝑠
ℎ
𝑎
𝑟
𝑒
𝑑
B_shared
), not Global Memory.

Since every active thread has work to do, every active thread must perform a binary search within the shared memory tile to find its local starting point.

Calculation:
Total Threads = 204,800.

Answer:
204,800 threads perform a binary search on shared memory.
