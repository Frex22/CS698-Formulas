# Assignment 2 — Submission Template
**Name:**  
**NJIT ID:**  

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
    # your code here
    # should return (B, P)
```

### Brief explanation

*Explain the key modification you made to the standard mergeSort.*

---

## Q4. A Theoretical Question

*Your answer here. Include a justification — a few lines of reasoning or a short derivation.*

*Example of how to write equations in Markdown:*

Inline equation: the answer is $f(n) = \log_2(\log_2 n)$

Display equation (centered on its own line):

$$\sqrt{\sqrt{n}} = n^{1/4} = n^{(1/2)^2}$$

---

## Q5. Beautiful Array (LeetCode)

### Solution

```python
def beautifulArray(n):
    # your code here
```

### LLM Usage

*Describe how you used the LLM. Options:*
- *Short summary of your prompting strategy and the key responses, or*
- *Copy-paste of the relevant part of the chat, or*
- *Link to the conversation:* [Chat link]()

---

## Q6. Maximum Sum Circular Subarray (LeetCode)

### Solution

```python
def maxSubarraySumCircular(nums):
    # your code here
```

### LLM Usage

*Describe how you used the LLM. Options:*
- *Short summary of your prompting strategy and the key responses, or*
- *Copy-paste of the relevant part of the chat, or*
- *Link to the conversation:* [Chat link]()

---
