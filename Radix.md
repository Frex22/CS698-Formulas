Here is the summary, cheat sheet, and optimization concepts for **Parallel Radix Sort**, followed by a working example of **Parallel Merge Sort**.

---

### Parallel Radix Sort Cheat Sheet (Single Iteration)

**The Goal:** Move all elements with a `0` at the current bit position to the front, and all elements with a `1` to the back, while maintaining their original relative order (Stability).

#### 1. The Variables
*   **$i$:** The unique index of the thread/element (0 to $N-1$).
*   **$b$:** The bit value of the element at the current position (0 or 1).
*   **`onesBefore`:** How many elements *before* index $i$ have a bit value of 1. (Calculated via **Exclusive Scan**).
*   **`totalOnes`:** The total number of 1s in the entire array.
*   **`totalZeros`:** $N - \text{totalOnes}$.

#### 2. The Formulas (Calculating Destination)
Every thread calculates where its element goes in the output array using these two logic paths:

| Condition | Destination Formula | Logic |
| :--- | :--- | :--- |
| **If bit $b == 0$** | $dest = i - \text{onesBefore}$ | The new index is the original index minus the space taken up by 1s that used to be in front of it. |
| **If bit $b == 1$** | $dest = \text{totalZeros} + \text{onesBefore}$ | The element must go *after* all the zeros. Its offset is determined by how many 1s were before it. |

#### 3. Step-by-Step Walkthrough Example
**Input:** `[5, 2, 6, 3]` (Binary: `101, 010, 110, 011`)
**Target:** Sort by Least Significant Bit (LSB).

1.  **Extract Bits ($b$):**
    *   5 (`...1`) $\rightarrow$ 1
    *   2 (`...0`) $\rightarrow$ 0
    *   6 (`...0`) $\rightarrow$ 0
    *   3 (`...1`) $\rightarrow$ 1
    *   **Bit Array:** `[1, 0, 0, 1]`

2.  **Exclusive Scan (Find `onesBefore`):**
    *   Input: `[1, 0, 0, 1]`
    *   Scan Result: `[0, 1, 1, 1]`
    *   *Note: The total sum (2) is stored at the end.* $\rightarrow$ `totalOnes = 2`, `totalZeros = 2`.

3.  **Calculate Destination:**
    *   **Thread 0 (Val 5, bit 1):** `totalZeros` + `onesBefore` $\rightarrow$ $2 + 0 = \mathbf{2}$
    *   **Thread 1 (Val 2, bit 0):** $i$ - `onesBefore` $\rightarrow$ $1 - 1 = \mathbf{0}$
    *   **Thread 2 (Val 6, bit 0):** $i$ - `onesBefore` $\rightarrow$ $2 - 1 = \mathbf{1}$
    *   **Thread 3 (Val 3, bit 1):** `totalZeros` + `onesBefore` $\rightarrow$ $2 + 1 = \mathbf{3}$

4.  **Write Output:**
    *   Index 0 gets Val 2
    *   Index 1 gets Val 6
    *   Index 2 gets Val 5
    *   Index 3 gets Val 3
    *   **Result:** `[2, 6, 5, 3]` (Evens first, then Odds. Stable).

---

### Optimization Concepts

#### 1. Optimizing for Memory Coalescing
*   **The Problem:** In the basic approach, Thread 0 might write to index 0 (a bucket for 0s), and Thread 1 might write to index 1000 (a bucket for 1s). This scattering destroys memory bandwidth performance.
*   **The Solution:** **Sort Locally First.**
    1.  Load a tile of data into **Shared Memory**.
    2.  Perform the Radix Sort/Shuffle inside Shared Memory.
    3.  Once sorted locally, the 0s are contiguous and the 1s are contiguous within the block.
    4.  The block writes its chunk of 0s to the global 0-bucket and its chunk of 1s to the global 1-bucket in coalesced, sequential writes.
*   **Requirement:** You need a global "Block Offset Scan" so Block 2 knows where its 0s start relative to Block 1's 0s.

#### 2. Choice of Radix Value (Bit-Width)
*   **The Concept:** Instead of sorting 1 bit at a time (Radix-2), sort 2 bits (Radix-4) or 4 bits (Radix-16) at a time.
*   **Trade-off:**
    *   **Pro (Higher Radix):** Fewer iterations. Sorting 32-bit integers with 1-bit radix takes 32 kernels. With 4-bit radix, it only takes 8 kernels.
    *   **Con (Higher Radix):** More buckets. A 4-bit radix needs 16 buckets. This splits the data into smaller chunks, making memory coalescing harder (shorter contiguous writes), and increases the size of the scan table needed to track bucket offsets.

#### 3. Thread Coarsening
*   **The Concept:** Assign **multiple elements** to a single thread (e.g., each thread processes 4 keys).
*   **Why it helps Coalescing:**
    *   If a block has more data elements but fewer threads, the "local buckets" inside that block become larger.
    *   Larger local buckets mean that when we write to Global Memory, we write larger contiguous chunks, which maximizes bandwidth.
    *   It also reduces the total number of blocks, which reduces the overhead of the secondary scan used to calculate block offsets.

---

### Parallel Merge Sort: A Working Example

Parallel Merge Sort combines the "Divide and Conquer" structure with the **Parallel Merge (Co-Rank)** logic we learned in Chapter 12.

**The Architecture:**
1.  **Local Sort:** Break array into small tiles. Each thread block sorts its tile independently (using a fast local sort like bitonic or radix).
2.  **Merge Tree:** Iteratively merge pairs of sorted lists until one big list remains.

**Working Example:**
*   **Input:** `[8, 2, 9, 4, 5, 3, 1, 6]` (Total 8 elements)

#### Step 1: Independent Block Sorting (Parallel)
Divide into chunks of 2.
*   Thread Block 0 sorts `[8, 2]` $\rightarrow$ `[2, 8]`
*   Thread Block 1 sorts `[9, 4]` $\rightarrow$ `[4, 9]`
*   Thread Block 2 sorts `[5, 3]` $\rightarrow$ `[3, 5]`
*   Thread Block 3 sorts `[1, 6]` $\rightarrow$ `[1, 6]`

#### Step 2: Parallel Merge (Stage 1)
Merge pairs of arrays.
*   **Merge A:** `[2, 8]` and `[4, 9]`
    *   Uses **Co-Rank** logic.
    *   Result: `[2, 4, 8, 9]`
*   **Merge B:** `[3, 5]` and `[1, 6]`
    *   Uses **Co-Rank** logic.
    *   Result: `[1, 3, 5, 6]`

#### Step 3: Parallel Merge (Stage 2 - Final)
Merge `A` (`[2, 4, 8, 9]`) and `B` (`[1, 3, 5, 6]`).
*   **Input Size:** 8 elements.
*   **Hardware:** 2 Threads.
*   **Job Split:** Each thread merges 4 output elements.

**Thread 0 Execution (Target indices 0-3):**
1.  **Co-Rank Search:** Find start for Rank 0 ($i=0, j=0$) and End for Rank 4.
    *   *Calculates split:* Take `[1, 3]` from B and `[2, 4]` from A.
2.  **Sequential Merge:**
    *   Merges `1, 2, 3, 4` into output slots `0, 1, 2, 3`.

**Thread 1 Execution (Target indices 4-7):**
1.  **Co-Rank Search:** Find start for Rank 4.
    *   *Calculates split:* Take `[8, 9]` from A and `[5, 6]` from B.
2.  **Sequential Merge:**
    *   Merges `5, 6, 8, 9` into output slots `4, 5, 6, 7`.

**Final Output:** `[1, 2, 3, 4, 5, 6, 8, 9]`
