# Assignment 2 — Submission
**Name:**  **Aakash**
**NJIT ID:** **31711346**

---

## Q1. Implementing a Max Heap

### What is wrong with the given implementation?

*Your explanation here. A few sentences is enough — describe the specific case(s) where the code breaks.*


Ideally looks like a an out of bounds problem, there is no check to see if a child exists or not so like in this part

c1_pos = 2*pos+1            # child 1 position

    c2_pos = 2*pos+2            # child 2 position
    
    if H[c1_pos] > H[c2_pos]:
    
      c_pos = c1_pos
      
    else:
    
      c_pos = c2_pos    
      
      do we even know that if c1_pos and c2_pos exist or index value for it exists if array is length 4 and if c1 = 5 and c2 = 6, then they dont exist in
      the array and in python we should get an "IndexError: list index out of range" error.
      so the solution can also be to check if c1 and c2 are in N if not we break and as the code or implementation desnot retain or remember values we start
      fresh in next iteration.

### Corrected implementation

```python
def heapMaxRemove(H):
    if len(H) == 0:
        return None
    
    max_val = H[0]              # save the max to return later
    x = H.pop()                 # pop last element
    
    if len(H) == 0:             # heap only had one element
        return max_val
    
    H[0] = x                    # put last element at root
    pos = 0
    n = len(H)
    
    while True:
        c1_pos = 2 * pos + 1
        c2_pos = 2 * pos + 2
        
        # Case 1: No children exist → leaf node, stop
        if c1_pos >= n:
            break
        
        # Case 2: Only left child exists
        elif c2_pos >= n:
            c_pos = c1_pos
        
        # Case 3: Both children exist, pick the bigger one
        else:
            if H[c1_pos] > H[c2_pos]:
                c_pos = c1_pos
            else:
                c_pos = c2_pos
        
        # Now do the swap check
        if H[pos] < H[c_pos]:
            H[pos], H[c_pos] = H[c_pos], H[pos]
            pos = c_pos
        else:
            break
    
    return max_val
    
```

---

## Q2. Finding the Minimum Element in a Max Heap

### Algorithm description

*Describe your algorithm in plain English. Which nodes do you need to examine, and why?*

So minimum element in a max heap will be in the last level of the tree, so if we compare from top and then reach the last level and compare laterally again 
it doesnt seem a good solution as many comparisions plus the approach itself doesnt look correct.
So we know that we need to get at the last level, so that means that any node with children will never be min so it has to be the leafnodes, now the idea is if we can
some how know that what index in a max heap is leaf nodes we can directly find a minimum from that in o(n)
so the idea in plain english is:

Identify the leaf nodes (indices n//2 through n-1)
Scan through all leaf nodes, tracking the smallest value
Return the smallest value


so I had to refer to this article to understand that the  indices (n//2 through n-1) will be leafnodes, so that seems to be the trick pretty much.
ref: [CSC228 – Module 8: Heap (JSU)](https://www.jsums.edu/nmeghanathan/files/2017/08/CSC228-Fall2017-Module-8-Heap.pdf)
### Implementation

```python
def heapFindMin(H):
    n = len(H)
    if n == 0:
        return None
    
    # Leaves occupy indices n//2 to n-1
    min_val = H[n // 2]
    
    for i in range(n // 2 + 1, n):
        if H[i] < min_val:
            min_val = H[i]
    
    return min_val
    
```

---

## Q3. Sorting with Original Position Memory

### Implementation

```python
def mergeSort(A):
    # Tag each element with its original position (1-indexed)
    tagged = [(A[i], i + 1) for i in range(len(A))]
    
    # Sort the tagged array
    sorted_tagged = mergeSortHelper(tagged)
    
    # Split into B (values) and P (positions)
    B = [pair[0] for pair in sorted_tagged]
    P = [pair[1] for pair in sorted_tagged]
    
    return B, P


def mergeSortHelper(arr):
    if len(arr) <= 1:
        return arr
    
    mid = len(arr) // 2
    left = mergeSortHelper(arr[:mid])
    right = mergeSortHelper(arr[mid:])
    
    return merge(left, right)


def merge(left, right):
    result = []
    i = 0
    j = 0
    
    while i < len(left) and j < len(right):
        if left[i][0] <= right[j][0]:
            result.append(left[i])
            i += 1
        else:
            result.append(right[j])
            j += 1
    
    while i < len(left):
        result.append(left[i])
        i += 1
    
    while j < len(right):
        result.append(right[j])
        j += 1
    
    return result
``` 
```

### Brief explanation

*Explain the key modification you made to the standard mergeSort.*
The idea is to track each element's original position throughout the sorting process. Before sorting begins, we pair each element with its original 1-indexed position, creating tagged tuples of the form (value, original_position). The merge sort algorithm then proceeds as normal, except comparisons are made only on the values (first element of each tuple). When two values are equal, we pick from the left subarray first to maintain stability. Since the position tag travels with the value through every split and merge, at the end we can separate the sorted tuples into two arrays: B containing the sorted values, and P containing the corresponding original positions.


---

## Q4. A Theoretical Question

*Your answer here. Include a justification — a few lines of reasoning or a short derivation.*

so the idea is very simple I am ie the function is changing by a constant value each time so here t=1/2

so by that idea if I press it k times it becomes n^{(1/2)^k} = n^{1/2^k}$$

now

We want to find the smallest $k$ such that:

$$n^{1/2^k} < 2$$

Taking $\log_2$ of both sides:

$$\frac{1}{2^k} \cdot \log_2(n) < 1$$

Rearranging:

$$\log_2(n) < 2^k$$

Taking $\log_2$ again:

$$\log_2(\log_2(n)) < k$$

Therefore the smallest integer $k$ satisfying this is:

$$k = \lceil \log_2(\log_2 n) \rceil$$

Intuition: Each square root operation halves the exponent of nn
n. So we're asking "how many times can we halve log⁡2(n)\log_2(n)
log2​(n) before it drops below 1?" — which is precisely log⁡2(log⁡2n)\log_2(\log_2 n)
log2​(log2​n).


*Example of how to write equations in Markdown:*

Inline equation: the answer is $f(n) = \log_2(\log_2 n)$

Display equation (centered on its own line):

$$\sqrt{\sqrt{n}} = n^{1/4} = n^{(1/2)^2}$$

---

## Q5. Beautiful Array (LeetCode) 

### Solution

```python
class Solution:
    def beautifulArray(self, n: int) -> list[int]:
        arr = [1]
        
        while len(arr) < n:
            temp = []
            
            # generate odds
            for x in arr:
                if 2 * x - 1 <= n:
                    temp.append(2 * x - 1)
            
            # generate evens
            for x in arr:
                if 2 * x <= n:
                    temp.append(2 * x)
            
            arr = temp
        
        return arr
```

### LLM Usage

*Describe how you used the LLM. Options:*
- *Short summary of your prompting strategy and the key responses, or*
- *Copy-paste of the relevant part of the chat, or*
- *Link to the conversation:* [Chat link](https://chatgpt.com/c/69ae2ff8-68e0-8325-ad2b-b571502fa9ab)

---

## Q6. Maximum Sum Circular Subarray (LeetCode)

### Solution

```python
class Solution:
    def maxSubarraySumCircular(self, nums: list[int]) -> int:
        total = sum(nums)
        
        # Kadane for max subarray
        curr_max = best_max = nums[0]
        # Kadane for min subarray
        curr_min = best_min = nums[0]
        
        for x in nums[1:]:
            curr_max = max(x, curr_max + x)
            best_max = max(best_max, curr_max)
            
            curr_min = min(x, curr_min + x)
            best_min = min(best_min, curr_min)
        
        # all negative case
        if best_max < 0:
            return best_max
        
        return max(best_max, total - best_min)
```

### LLM Usage

*Describe how you used the LLM. Options:*
- *Short summary of your prompting strategy and the key responses, or*
- *Copy-paste of the relevant part of the chat, or*
- *Link to the conversation:* [Chat link](https://chatgpt.com/c/69ae3584-6128-832e-b5b2-7b25ac42bedd)

---
