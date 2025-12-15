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
