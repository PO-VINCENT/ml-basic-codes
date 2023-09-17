"""一,双指针
时间复杂度一般是O(n)
Slicing Windows and two points
什么是滑动窗口？
其实就是一个队列,比如例题中的 abcabcbb，进入这个队列（窗口）为 abc 满足题目要求，当再进入 a，队列变成了 abca，这时候不满足要求。
所以，我们要移动这个队列！
如何移动？
我们只要把队列的左边的元素移出就行了，直到满足题目要求！
一直维持这样的队列，找出队列出现最长的长度时候，求出解！
时间复杂度：O(n)
1) 经典双指针使用统计非重复数量
"""
#3. Longest Substring Without Repeating Characters，Given a string, find the length of the
#longest substring without repeating characters
class Solution:
    def lengthOfLongestSubstring(self, s):
        if not s:
            return 0
        n=len(s)
        left, right = 0, 0
        max_len = 0
        visited = set([])
        for left in range(n):
            while right <n and s[right] not in visited:
                visited.add(s[right])
                right+=1
            max_len=max(max_len,right-left)
            visited.discard(s[left])
        return max_len
"""
2) 同一个数组，要in-place的去掉里面的特殊数，类似0，或者重复
Step1: 设置双指针，同一起点，注意如果有特殊需求，类似不可出现两次这样的，起点设置不一定是从0开始
Step2: 让右指针作为大循环，然后在里面先判断不满足条件的情况，让左右指针的值互换，然后左指针前行，为小循环
Step3: 返回值，注意返回值是否加上1
"""
#283. Move Zeroes，Given an array nums, write a function to move all 0's to the end of it
# while maintaining the relative order of the non-zero elements.
class Solution:
    def moveZeros(self, nums):
        left, right = 0, 0
        n = len(nums)
        while right < n:
            # right和left同一起点，如果是非零元素，他们都会同时走，有left+=1，也有right+=1，只有当出现0元素，right会先走一步，跳过0，left进入0
            if nums[right] !=0:
                if left!=right:
                    nums[left] = nums[right]
                left +=1
            right +=1
        while left < n:
            if nums[left] !=0:
               nums[left] = 0
            left +=1
        return nums
# 604, Window Sum. Given an array of n integers, and a moving window(size k), move the window at each",
# iteration from the start of the array, find the sum of the element inside the window at",
# each moving
class Solution:
    def winSum(self,nums,k):
        if not nums or k <0:
            return []
        n = len(nums)
        left, right = 0, k-1
        res = []
        while right < n:
            if left == 0:
                sum_each = sum(nums[0:right+1])
                res.append(sum_each)
            else:
                sum_each = res[-1] - nums[left-1]+nums[right]
                res.append(sum_each)
            left+=1
            right+=1
        return res
#287. Find the Duplicate Number。 Given an array nums containing n + 1 integers where each integer is
# between 1 and n (inclusive), prove that at least one duplicate number must exist. Assume that there
# is only one duplicate number, find the duplicate one
"""使用类似环形链表142题的方式，
关键是理解如何把输入的数组看作为链表。首先要明确前提，整个数组中的nums是在[1,n]之间，考虑两种情况：
1) 如果数组中没有重复的数字，以数组[1,3.4.2]为例，我们将数组下标n和数nums[n]建立映射关系f(n), 其隐射关系n->f(n)为：
0->1 
1->3
2->4
3->2
我们以下标0出发，根据f(n)计算出一个值，以这个值为新的下标，再用这个函数计算，以此类推，直到下标超界。这样就可以产生一个类似链表的
序列0->1->3->2->4->null
2) 如果数组中有重复的数字，以数组[1,3.4.2，2]为例，我们将数组下标n和数nums[n]建立映射关系f(n), 其隐射关系n->f(n)为：
0->1 
1->3
2->4
3->2
4->2
同样可以构造链表：0->1->3->2->4->3->2->4->2，从理论上讲，数组中如果有重复的数，那么就会产生多对一的映射，这样，形成的链表就一定会有环路了
综述，如果数组中有重复数字，那么链表中就存在环，找到环的入口就是找到这个数字
142中慢指针走一步 slow=slow.next 等于本题中slow=nums[slow]
142中快指针走一步 fast=fast.next 等于本题中fast=nums[nums[fast]]
"""
class Solutions:
    def findDulicate(self,nums):
        if len(nums)<=1:
            return -1
        slow = nums[0]
        fast = nums[nums[0]]
        while slow!=fast:
            slow = nums[slow]
            fast= nums[nums[fast]]
        fast = 0
        while fast!=slow:
            fast = nums[fast]
            slow = nums[slow]
        return slow
# 142. Linked List Cycle II.Given a linked list, return the node where the cycle begins.
# If there is no cycle, return null.To represent a cycle in the given linked list, we use
# an integer pos which represents the position (0-indexed) in the linked list where tail
# connects to. If pos is -1, then there is no cycle in the linked list. Note: Do not modify the linked list
class Solutions:
    def detectCycle(self, head):
        if not head or head.next:
            return -1
        slow, fast = head, head
        while fast and fast.next:
            slow = slow.next
            fast = fast.next.next
            if slow is fast:
                break
        if fast is slow:
            slow = head
            while fast is not slow:
                fast = fast.next
                slow = slow.next
            return slow
        return None
# 2. Add Two Numbers. You are given two non-empty linked lists representing two non-negative integers.
# The digits are stored in reverse order and each of their nodes contain a single digit. Add the two numbers and return it as a linked list.",
class Solutions:
    def addTwoNumbers(self,l1:ListNode,l2:ListNode):
        #首先创建一个虚拟节点，并创建一个current指针，指向这个节点
        current = dummy = ListNode()
        #初始化carry和两个链表对应节点相加的值
        carry, value = 0, 0
        #下面的while循环中之所以有carry，是为了处理两个链表最后节点相加出现进位的情况
        #当两个节点都走完而且最后的运算并没有进位时，就不会进入这个循环
        while carry or l1 or l2:
            #让value先等于carry既有利于下面两个if语句中两个对应节点值相加，
            # 也是为了要处理两个链表最后节点相加出现进位的情况
            value = carry
            #只要其中一个链表没走完，就需要计算value的值
            #如果其中一个链表走完，那么下面的计算就是加总carry和其中一个节点的值
            #如果两个链表都没走完，那么下面的计算就是carry+对应的两个节点的值
            if l1: l1, value = l1.next, l1.val + value
            if l2: l2, value = l2.next, l2.val + value
            #为了防止value值大于十，出现进位，需要特殊处理
            #如果value小于十，下面这行的操作对于carry和value的值都没有影响
            carry, value = divmod(value, 10)
            #利用value的值创建一个链表节点，并让current.next指向它
            current.next = ListNode(value)
            #移动current指针到下一个节点
            current = current.next
        #最后只要返回dummy的下一个节点就是我们想要的答案。
        return dummy.next
# 剑指 Offer 59 - I. 滑动窗口的最大值: 给定一个数组 nums 和滑动窗口的大小 k，请找出所有滑动窗口里的最大值。
# 基本思想：
# 维护一个长度小于等于k的单调递减的单调队列，不断向右滑动，该单调队列的第一个元素即为该窗口的最大元素
#
class Solution:
    def maxInWindows(self, num, size):
""""
二，BFS
1, 齐头并进的广度优先遍历问题：
(1) 树的宽度优先遍历
"""
# 102. Binary Tree Level Order Traversal
# Given a binary tree, return the level order traversal of its nodes' values. (ie, from left to right, level by level
class TreeNode:
    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None
from collections import deque
class Solution:
    def levelOrder(self,root:TreeNode) -> List[List[int]]:
        if not root:
            return []
        queue = deque([root])
        res =[]
        while queue:
            level =[]
            for _ in range(len(queue)):
                node = queue.popleft()
                level.append(node.val)
                if node.left:
                    queue.append(node.left)
                if node.right:
                    queue.append(node.right)
            res.append(level)
        return res
# 107. Binary Tree Level Order Traversal II
# Given a binary tree, return the bottom-up level order traversal of its nodes' values. (ie, from left to right, level by level from leaf to right
class TreeNode:
    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None
from collections import deque
class Solution:
    def levelOrderBottom(self, root:TreeNode) -> List[List[int]]:
        if not root:
            return []
        queue = deque([root])
        res = []
        while queue:
            level = []
            for _ in range(len(queue)):
                node = queue.popleft()
                level.append(node.val)
                if node.left:
                    queue.append(node.left)
                if node.right:
                    queue.append(node.right)
            res.insert(0,level)
        return res
# 103. Binary Tree Zigzag Level Order Traversal
# Given a binary tree, return the zigzag level order traversal of its nodes' values. (ie, from left to right, then right to left for the next level and alternate between)
class TreeNode:
    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None
from collections import deque
class Solution:
    def zigzagLevelOrder(self, root: TreeNode) -> List[List[int]]:
        if not root:
            return []
        queue = deque([root])
        count =0
        res = []
        while queue:
            count +=1
            level =[]
            for _ in range(len(queue)):
                node = queue.popleft()
                if count%2 ==1:
                    level.append(node.val)
                else:
                    level.insert(0,node.val)
                if node.left:
                    queue.append(node.left)
                if node.right:
                    queue.append(node.right)
            res.append(level)
        return res
"""
（2）无权图的最短路径
--依赖的数据结构：队列
--应用：无权图中找到最短路径
--注意事项：无权图中遍历，加入队列之后，必须马上标记【已经访问】
在图中，由于 图中存在环，和深度优先遍历一样，广度优先遍历也需要在遍历的时候记录已经遍历过的结点。
特别注意：将结点添加到队列以后，一定要马上标记为「已经访问」，否则相同结点会重复入队，
这一点在初学的时候很容易忽略。如果很难理解这样做的必要性，建议大家在代码中打印出队列中的元素进行调试：
在图中，如果入队的时候不马上标记为「已访问」，相同的结点会重复入队，这是不对的。另外一点还需要强调，
广度优先遍历用于求解「无权图」的最短路径，因此一定要认清「无权图」这个前提条件。
如果是带权图，就需要使用相应的专门的算法去解决它们
"""
# 323. 无向图中连通分量的数目
# 给定编号从 0 到 n-1 的 n 个节点和一个无向边列表（每条边都是一对节点），请编写一个函数来计算无向图中连通分量的数目。
"""BFS"""
class Solution:
    def countComponents(self, n: int, edges: List[List[int]]) -> int:
        from collections import defaultdict, deque
        graph = defaultdict(list)
        #只有当list里面是两个元素才可以使用这样的方式
        for x, y in edges:
            graph[x].append(y)
            graph[y].append(x)
        visited = set()
        connected = 0
        def bfs(i):
            queue = deque([i])
            while queue:
                i = queue.pop()
                for j in graph[i]:
                    if j not in visited:
                        visited.add(j)
                        queue.append(j)
        for i in range(n):
            if i not in visited:
                connected+=1
                visited.add(i)
                bfs(i)
        return connected
"""DFS"""
class Solution:
    def countComponents(self, n: int, edges: List[List[int]]) -> int:
        from collections import defaultdict, deque
        graph = defaultdict(list)
        for x, y in edges:
            graph[x].append(y)
            graph[y].append(x)
        visited = set()
        connected = 0
        def dfs(i):
            visited.add(i)
            for j in graph[i]:
                if j not in visited:
                    dfs(j)
        for i in range(n):
            if i not in visited:
                connected+=1
                dfs(i)
        return connected
"""
2, 二维平面上的搜索问题
"""
#695，岛屿的最大面积
#给定一个包含了一些0和1的费控二维数组grid,一个岛屿是由一些相邻的1（代表土地）
#构成的组合，这里的相邻要求两个1必须在水平或者竖直方向上相邻，可以假设grid的
#四个边缘都被0（代表水）包围着，找到给定的二维数组中最大的岛屿面积（如果没有返回0）
"""BFS"""
class Solution:
    def maxAreaOfIsland(self, grid: List[List[int]]) -> int:
        from collections import deque
        ans = 0
        for i, l in enumerate(grid):
            for j,n in enumerate(l):
                cur = 0
                q = deque([(i,j)])
                while q:
                    cur_i, cur_j = q.popleft()
                    # put all lands into queue and pass water
                    if cur_i<0 or cur_j<0 or cur_i == len(grid) or cur_j ==len(grid[0]) or grid[cur_i][cur_j] !=1:
                        continue
                    cur +=1
                    grid[cur_i][cur_j] = 0
                    for di, dj in [[0,1],[0,-1],[1,0],[-1,0]]:
                        next_i, next_j = cur_i+di, cur_j+dj
                        q.append((next_i,next_j))
                ans = max(ans,cur)
        return ans
"""DFS"""
# Time complex: O(R*C),R是给定的网格中的行数, C是列数，我们访问每个网格最
# Space complex: O(R*C),
class Solution:
    def maxAreaOfIsland(self, grid: List[List[int]]) -> int:
        from collections import deque
        ans = 0
        def dfs(self,grid,cur_i,cur_j):
            if cur_i < 0 or cur_j < 0 or cur_i == len(grid) or cur_j == len(grid[0]) or grid[cur_i][cur_j] != 1:
                return 0
            grid[cur_i][cur_j] = 0
            ans =1
            for di, dj in [[0,1],[0,-1],[1,0],[-1,0]]:
                next_i, next_j = cur_i + di, cur_j + dj
                ans += self.dfs(grid,next_i,next_j)
            return ans
        for i, l in enumerate(grid):
            for j,n in enumerate(l):
                ans = max(self.dfs(grid,i,j),ans)
        return ans
""""
三，排序
1, 时间复杂度 O(n^2) 级排序算法
1）冒泡排序
是入门级的算法，但也有一些有趣的玩法。通常来说，冒泡排序有三种写法：
一边比较一边向后两两交换，将最大值 / 最小值冒泡到最后一位；
经过优化的写法：使用一个变量记录当前轮次的比较是否发生过交换，如果没有发生交换表示已经有序，不再继续排序；
进一步优化的写法：除了使用变量记录当前轮次是否发生交换外，再使用一个变量记录上次发生交换的位置，下一轮排序时到达上次交换的位置就停止比较。
2, 时间复杂度 O(nlogn) 级排序算法
1）希尔排序
虽然原始的希尔排序最坏时间复杂度仍然是 O(n^2)，但经过优化的希尔排序可以达到 O(n^{1.3}) 甚至 O(n^{7/6})
希尔排序本质上是对插入排序的一种优化，它利用了插入排序的简单，又克服了插入排序每次只交换相邻两个元素的缺点。它的基本思想是：
将待排序数组按照一定的间隔分为多个子数组，每组分别进行插入排序。这里按照间隔分组指的不是取连续的一段数组，而是每跳跃一定间隔取一个值组成一组
逐渐缩小间隔进行下一轮排序
最后一轮时，取间隔为 11，也就相当于直接使用插入排序。但这时经过前面的「宏观调控」，数组已经基本有序了，所以此时的插入排序只需进行少量交换便可完成
2）堆排序
堆：符合以下两个条件之一的完全二叉树：
根节点的值 ≥ 子节点的值，这样的堆被称之为最大堆，或大顶堆；
根节点的值 ≤ 子节点的值，这样的堆被称之为最小堆，或小顶堆。
堆排序过程如下：
用数列构建出一个大顶堆，取出堆顶的数字；
调整剩余的数字，构建出新的大顶堆，再次取出堆顶的数字；
循环往复，完成整个排序。
整体的思路就是这么简单，我们需要解决的问题有两个：
如何用数列构建出一个大顶堆；
取出堆顶的数字后，如何将剩余的数字调整成新的大顶堆。
构建大顶堆 & 调整堆
构建大顶堆有两种方式：
方案一：从 0 开始，将每个数字依次插入堆中，一边插入，一边调整堆的结构，使其满足大顶堆的要求；
方案二：将整个数列的初始状态视作一棵完全二叉树，自底向上调整树的结构，使其满足大顶堆的要求。
第二种方案更加常用。
在介绍堆排序具体实现之前，我们先要了解完全二叉树的几个性质。将根节点的下标视为 0，则完全二叉树有如下性质：
假设一个二叉树的深度为h，除第 h 层外，其它各层 (1～h-1) 的结点数都达到最大个数，第 h 层所有的结点都连续集中在最左边，这就是完全二叉树
对于完全二叉树中的第 i 个数，它的左子节点下标：left = 2i + 1
对于完全二叉树中的第 i 个数，它的右子节点下标：right = left + 1
对于有 n 个元素的完全二叉树(n≥2)(n≥2)，它的最后一个非叶子结点的下标：n/2 - 1
"""
def heapSort(arr):
    # 构建初始大顶堆
    build_max_heap(arr)
    lenth = len(arr)
    for i in range(lenth-1, 0, -1):
        # 将最大值交换到数组最后
        swap(arr,0,i)
        # 调整剩余数组，使其满足大顶堆
        max_heapify(arr,0,i)
# 桂建初始大顶堆
def build_max_heap(arr):
    n = len(arr)
    for i in range(n/2-1,0,-1):
        max_heapify(arr,i,n)
# 调整大顶堆，第三个参数表示剩余未排序的数量，也是剩余堆的大小
def max_heapify(arr, i, heap_size):
    # 左子节点下标
    l = 2*i+1
    # 右子节点下标
    r = l+1
    largest = i
    # 与左子树节点比较
    if heap_size >1 and arr[l]>arr[largest]:
        largest = l
    # 与右子树节点比较
    if heap_size >1 and arr[r]>arr[largest]:
        largest = r
    if largest != i:
        # 将最大值交换为根节点
        swap(arr, i, largest)
        # 再次调整交换后的最大顶堆
        max_heapify(arr,largest,heap_size)
def swap(arr, i, j):
    temp = arr[i]
    arr[i] = arr[j]
    arr[j] = arr[i]
"""2, 时间复杂度 O(nlogn) 级排序算法
3）快排
时间复杂度O(nlogn)，时间复杂度也是O(nlogn)
基本思想：
step1: 从数组中取出一个数，称之为基数(pivot)
step2: 遍历数组，将比基数大的数字放在其右边，比基数小的放在左边，遍历完成，数组被分成左右两个区域
step3: 将两个区域视为两个数组，重复前面两个步骤，直到完成为止
"""
class Solution:
    def sortArray(self, nums: List[int]) -> List[int]:
        if not nums or len(nums) == 0:
            return
        self.quickSort(nums, 0, len(nums) - 1)
        return nums
    def quickSort(self, nums, start, end):
        if start >= end:
            return
        left, right = start, end
        pivot = (nums[left] + nums[right]) / 2
        while left <= right:
            while left <= right and nums[left] < pivot:
                left += 1
            while left <= right and nums[right] > pivot:
                right -= 1
            if left <= right:
                nums[right], nums[left] = nums[left], nums[right]
                left += 1
                right -= 1
        self.quickSort(nums, start, right)
        self.quickSort(nums, left, end)
#347, Top K Frequent Elements
class Solution:
    def findKthLargest(self, nums: List[int], k: int) -> int:
        if not nums or len(nums) == 0:
            return
        self.qucikSelect(nums, 0, len(nums) - 1, k)
        return nums[k - 1]
    def qucikSelect(self, nums, start, end, k):
        if start >= end:
            return
        left, right = start, end
        pivot = (nums[left] + nums[right]) / 2
        while left <= right:
            while left <= right and nums[left] > pivot:
                left += 1
            while left <= right and nums[right] < pivot:
                right -= 1
            if left <= right:
                nums[right], nums[left] = nums[left], nums[right]
                left += 1
                right -= 1
        if start + k - 1 <= right:
            self.qucikSelect(nums, start, right, k)
        if start + k - 1 >= left:
            self.qucikSelect(nums, left, end, k - (left - start))
""""
四，二叉树
二叉树的遍历问题或者其他的任何问题，都考虑把它落实在每个子树上，然后在每颗字数上考虑，推广到全局
1，二叉树的前/中/后序遍历问题
"""
# 144. Binary Tree Preorder Traversal。 Given a binary tree, return the preorder traversal of its nodes' values
class TreeNode:
    def __init__(self,x):
        self.val = x
        self.left = None
        self.right = None
"""非递归的方式--使用stack的方法：因为栈先进后出，可以pop右边的能力"""
class Solution:
    def preorderTraversal(self,root:TreeNode): # root,left,right
        if not root:
            return []
        stack = [root]
        res = []
        while stack:
            node = stack.pop()
            res.append(node.val)
            if node.right:
                stack.append(node.right)
            if node.left:
                stack.append(node.left)
        return res
"""递归的方式--使用递归方法"""
class Solution:
    def preorderTraversal(self,root:TreeNode): # root,left,right
        if not root:
            return []
        res = []
        self.traverse(root,res)
        return res
    def traverse(self,root,res):
        if not root:
            return
        res.append(root.val)
        self.traverse(root.left,res)
        self.traverse(root.right,res)
# 94. Binary Tree Inorder Traversal.  Given a binary tree, return the inorder traversal of its nodes' values
"""递归的方式--使用递归方法"""
class Solution:
    def inorderTraversal(self,root:TreeNode): # left,root,right
        if not root:
            return []
        res = []
        self.traverse(root,res)
        return res
    def traverse(self,root,res):
        if not root:
            return
        self.traverse(root.left,res)
        res.append(root.val)
        self.traverse(root.right,res)
"""非递归的方式--使用stack的方式"""
class Solution:
    def inorderTraversal(self,root:TreeNode): # left,root,right
        if not root:
            return []
        stack = []
        res = []
        while root:
            res.append(root)
            root=root.left
        while stack:
            node =stack.pop()
            res.append(node.val)
            if node.right:
                node1 = node.right
                while node1:
                    stack.append(node1)
                    node1 = node1.left
        return res
#144. Binary Tree Preorder Traversal.Given a binary tree, return the preorder traversal of its nodes' values
"""递归的方式--使用递归方法"""
class Solution:
    def postorderTraversal(self, root: TreeNode):# left,right,root
        if not root:
            return []
        res = []
        self.traverse(root, res)
        return res
    def traverse(self, root, res):
        if not root:
            return
        self.traverse(root.left, res)
        self.traverse(root.right, res)
        res.append(root.val)
#105. Construct Binary Tree from Preorder and Inorder Traversal
class Solution:
    def reConstructBinaryTree(self, pre, tin):
        if not pre or not tin:
            return None
        rootVal = pre[0]
        id = tin.index(pre[0])
        root = TreeNode(rootVal)
        root.left = self.reConstructBinaryTree(pre[1:id+1],tin[:id])
        root.right = self.reConstructBinaryTree(pre[id+1:],tin[id+1:])
        return root
""""
四，二叉树的基本变化序
1, DFS遍历，一般不需要返回值。但是可以在遍历的过程中进行各种操作，以达到目的
"""
# 226. Invert Binary Tree
class TreeNode:
    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None
class Solution:
    def invertTree(self,root):
        if not root:
            return None
        self.dfs(root)
        return root
    def dfs(self,root):
        if not root:
            return
        left =root.left
        right = root.right
        root.left = right
        root.right = left
        if root.right:
            self.dfs(root.right)
        if root.left:
            self.dfs(root.left)
""""
2，分治法：分治法一般要返回值，为了求最大值，需要问题转化成解决左子树上深度，
和右子树深度。然后合并中最大的就是全局深度。因为每次遍历深度加一。
"""
#101. Symmetric Tree。Given a binary tree, check whether it is
# a mirror of itself (ie, symmetric around its center。
class TreeNode:
    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None
class Solution:
    def isSymmetric(self, root: TreeNode):
        if not root:
            return True
        return self.checkSymmetric(root.left,root.right)
    def checkSymmetric(self,nodeA,nodeB):
        if nodeA is None or nodeB is None:
            return False
        if nodeA is None and nodeB is None:
            return True
        if nodeA.val != nodeB.val:
            return False
        inner_res = self.checkSymmetric(nodeA.left,nodeB.right)
        outer_res = self.checkSymmetric(nodeA.right, nodeB.left)
        return inner_res and outer_res
""""
3，Path Sum类型问题一般分为三种方式:
1) 判断是否存在从root- leaf 和为指定的数字的路径
（1）构造一个递归出口，一般就是在root 为空时候
（2）构造一个到leaf节点时候 找到指定路径的情况，
    一般就是，当root.left和root.right为空，root.val == 此时target数字
（3）然后进行左右递归，注意，参数中的，root和target数字都是要需要更新的
"""
#112. Path Sum. Given a binary tree and a sum, determine if the tree has a root-to-leaf path such that
# adding up all the values along the path equals the given sum.
class TreeNode:
    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None
class Solution:
    def hasPathSum(self,root,sum):
        if root is None:
            return False
        if root.left is None and root.left is None:
            return sum == root.val
        return self.hasPathSum(root.left, sum-root.val) or self.hasPathSum(root.right, sum-root.val)
#437. Path Sum III. You are given a binary tree in which each node contains an integer value
#Find the number of paths that sum to a given value.The path does not need to start or end at
#the root or a leaf, but it must go downwards (traveling only from parent nodes to child nodes)
#The tree has no more than 1,000 nodes and the values are in the range -1,000,000 to 1,000,000
class TreeNode:
    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None
class Solution:
    def pathSum(self, root: TreeNode, sum: int):
 # 思路分析： 因为题目中要求，只要path之和等于所给的sum，可以不必须是从root开始，而是可以从任何地方开始。
 # 那么就需要考虑，root, root.left, root.right 三种情况下的。总的path的数量就是他们三个所包含的之和
        if not root:
            return 0
        res = []
        self.helper(root, [], sum, res)
        return res
    def helper(self, root, path, target, res):
        if not root:
            return
        path += [root.val]
        flag = (root.left == None and root.right === None)
        if root.val ==target and flag:
            res.append(path[:])
            return
        if root.left:
            self.helper(root.left, path[:], target-root.val,res)
        if root.right:
            self.helper(root.right, path[:], target-root.val,res)
        path.pop()
"""
4，关于最大最小长度问题：一般采用的是分治法
(1) 一般需要返回三类值:1)curLen/curSum, 返回当前所求的长度和值；2）maxLen/maxSum，返回最值; 3) node或者nodeVal,
返回节点或者节点数值，具体选择哪一个，看题中要求；
(2) 对于空集的判断if not root,需要注意的是，最值在初始化的时候，最大值一般是用-sys.maxsizem,最小值是用sys.maxsize
(3) divide，分别对左右进行循环
(4) conque, 在conque的时候，需要分两步:1)先根据调解，求出curLen/curSum；2）然后在判断出最值
"""
# Minimum Subtree.Given a binary tree, find the subtree with minimum sum. Return the root of the subtree
class TreeNode:
    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None
class Solution:
    def findSubtree(self, root: TreeNode):
        if not root:
            return None
        node, minSubtree,Sum = self.helper(root)
        return node
    def helper(self,root):
        if not root:
            return None, 0, sys.maxsize
        left_node, left_min_sum, left_sum = self.helper(root.left)
        right_node, right_min_sum, right_sum = self.helper(root.right)
        total_sum = left_sum + root.val+ right_sum
        if left_min_sum == min(left_min_sum, right_min_sum, total_sum):
            return left_node, left_min_sum, total_sum
        if right_min_sum == min(left_min_sum, right_min_sum, total_sum):
            return right_node , right_min_sum, total_sum
        if total_sum == min(left_min_sum, right_min_sum, total_sum):
            return root, total_sum, total_sum
# 298. Binary Tree Longest Consecutive Sequence.Given a binary tree, find the length of the longest consecutive sequence path
# The path refers to any sequence of nodes from some starting node to any node in the tree along the parent-child connections.
# The longest consecutive path need to be from parent to child (cannot be the reverse)
class TreeNode:
    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None
class Solution:
    def longestConsecutive(self, root: TreeNode):
        if not root:
            return 0
        maxLen, curLen, curVal = self.helper(root)
        return maxLen
    def helper(self, root):
        if not root:
            return -sys.maxsize, 0, 0
        left_maxLen, left_curLen, left_curVal = self.helper(root.left)
        right_maxLen, right_curLen, right_curVal = self.helper(root.right)
        curLen =1
        if root.val == left_curVal -1:
            curLen = max(curLen, left_curLen+1)
        if root.val == right_curVal -1:
            curLen = max(curLen, right_curLen+1)
        if left_maxLen == max(curLen, left_maxLen, right_maxLen):
            return left_maxLen, curLen, root.val
        if right_maxLen == max(curLen, left_maxLen, right_maxLen):
            return right_maxLen, curLen, root.val
        return curLen,curLen,root.val
"""
5, 二叉树宽度有限搜索BFS：主要是使用deque这类函数先进先出的优点，配合deque.popleft(), 实现层序遍历
"""
# 102. Binary Tree Level Order Traversal。Given a binary tree, return the level order traversal
# of its nodes' values. (ie, from left to right, level by level)
class TreeNode:
    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None
class Solution:
    def levelOrder(self, root: TreeNode):
        if not root:
            return []
        queue = deque([root])
        res = []
        while queue:
            level = []
            for _ in range(len(queue)):
                node = queue.popleft()
                level.append(node.val)
                if node.left:
                    queue.append(node.left)
                if node.right:
                    queue.append(node.right)
            res.append(level)
        return res
"""一,双指针
时间复杂度一般是O(n)
Slicing Windows and two points
什么是滑动窗口？
其实就是一个队列,比如例题中的 abcabcbb，进入这个队列（窗口）为 abc 满足题目要求，当再进入 a，队列变成了 abca，这时候不满足要求。
所以，我们要移动这个队列！
如何移动？
我们只要把队列的左边的元素移出就行了，直到满足题目要求！
一直维持这样的队列，找出队列出现最长的长度时候，求出解！
时间复杂度：O(n)
1) 经典双指针使用统计非重复数量
"""
#3. Longest Substring Without Repeating Characters，Given a string, find the length of the
#longest substring without repeating characters
class Solution:
    def lengthOfLongestSubstring(self, s):
        if not s:
            return 0
        left = 0
        # the left boundary of window
        curLen, maxLen = 0, 0
        visited = set()
        n = len(s)
        for i in range(n):
            curLen += 1
            while s[i] in visited:
                visited.remove(s[left])
                curLen -= 1
                left += 1
            if maxLen < curLen:
                maxLen = curLen
            visited.add(s[i])
        return maxLen
"""
2) 同一个数组，要in-place的去掉里面的特殊数，类似0，或者重复
Step1: 设置双指针，同一起点，注意如果有特殊需求，类似不可出现两次这样的，起点设置不一定是从0开始
Step2: 让右指针作为大循环，然后在里面先判断不满足条件的情况，让左右指针的值互换，然后左指针前行，为小循环
Step3: 返回值，注意返回值是否加上1
"""
#283. Move Zeroes，Given an array nums, write a function to move all 0's to the end of it
# while maintaining the relative order of the non-zero elements.
class Solution:
    def moveZeros(self, nums):
        left, right = 0, 0
        n = len(nums)
        while right < n:
            if nums[right] !=0:
                if left!=right:
                    nums[left] = nums[right]
                left +=1
            right +=1
        while left < n:
            if nums[left] !=0:
               nums[left] = 0
            left +=1
        return nums
# 604, Window Sum. Given an array of n integers, and a moving window(size k), move the window at each",
# iteration from the start of the array, find the sum of the element inside the window at",
# each moving
class Solution:
    def winSum(self,nums,k):
        if not nums or k <0:
            return []
        left, right = 0, k-1
        res = []
        while right < 0:
            if left == 0:
                sum_each = sum(nums[0:right+1])
                res.append(sum_each)
            else:
                sum_each = res[-1] - nums[left-1]+nums[right]
                res.append(sum_each)
            left+=1
            right+=1
        return res
#287. Find the Duplicate Number。 Given an array nums containing n + 1 integers where each integer is
# between 1 and n (inclusive), prove that at least one duplicate number must exist. Assume that there
# is only one duplicate number, find the duplicate one
"""使用类似环形链表142题的方式，
关键是理解如何把输入的数组看作为链表。首先要明确前提，整个数组中的nums是在[1,n]之间，考虑两种情况：
1) 如果数组中没有重复的数字，以数组[1,3.4.2]为例，我们将数组下标n和数nums[n]建立映射关系f(n), 其隐射关系n->f(n)为：
0->1 
1->3
2->4
3->2
我们以下标0出发，根据f(n)计算出一个值，以这个值为新的下标，再用这个函数计算，以此类推，直到下标超界。这样就可以产生一个类似链表的
序列0->1->3->2->4->null
2) 如果数组中有重复的数字，以数组[1,3.4.2，2]为例，我们将数组下标n和数nums[n]建立映射关系f(n), 其隐射关系n->f(n)为：
0->1 
1->3
2->4
3->2
4->2
同样可以构造链表：0->1->3->2->4->3->2->4->2，从理论上讲，数组中如果有重复的数，那么就会产生多对一的映射，这样，形成的链表就一定会有环路了
综述，如果数组中有重复数字，那么链表中就存在环，找到环的入口就是找到这个数字
142中慢指针走一步 slow=slow.next 等于本题中slow=nums[slow]
142中快指针走一步 fast=fast.next 等于本题中fast=nums[nums[fast]]
"""
class Solutions:
    def findDulicate(self,nums):
        if len(nums)<=1:
            return -1
        slow = nums[0]
        fast = nums[nums[0]]
        while slow!=fast:
            slow = nums[slow]
            fast= nums[nums[fast]]
        fast = 0
        while fast!=slow:
            fast = nums[fast]
            slow = nums[slow]
        return slow
# 142. Linked List Cycle II.Given a linked list, return the node where the cycle begins.
# If there is no cycle, return null.To represent a cycle in the given linked list, we use
# an integer pos which represents the position (0-indexed) in the linked list where tail
# connects to. If pos is -1, then there is no cycle in the linked list. Note: Do not modify the linked list
class Solutions:
    def detectCycle(self, head):
        if not head or head.next:
            return -1
        slow, fast = head, head
        while fast and fast.next:
            slow = slow.next
            fast = fast.next.next
            if slow is fast:
                break
        if fast is slow:
            slow = head
            while fast is not slow:
                fast = fast.next
                slow = slow.next
            return slow
        return None
# 2. Add Two Numbers. You are given two non-empty linked lists representing two non-negative integers.
# The digits are stored in reverse order and each of their nodes contain a single digit. Add the two numbers and return it as a linked list.",
class Solutions:
    def addTwoNumbers(self,l1:ListNode,l2:ListNode):
        #首先创建一个虚拟节点，并创建一个current指针，指向这个节点
        current = dummy = ListNode()
        #初始化carry和两个链表对应节点相加的值
        carry, value = 0, 0
        #下面的while循环中之所以有carry，是为了处理两个链表最后节点相加出现进位的情况
        #当两个节点都走完而且最后的运算并没有进位时，就不会进入这个循环
        while carry or l1 or l2:
            #让value先等于carry既有利于下面两个if语句中两个对应节点值相加，
            # 也是为了要处理两个链表最后节点相加出现进位的情况
            value = carry
            #只要其中一个链表没走完，就需要计算value的值
            #如果其中一个链表走完，那么下面的计算就是加总carry和其中一个节点的值
            #如果两个链表都没走完，那么下面的计算就是carry+对应的两个节点的值
            if l1: l1, value = l1.next, l1.val + value
            if l2: l2, value = l2.next, l2.val + value
            #为了防止value值大于十，出现进位，需要特殊处理
            #如果value小于十，下面这行的操作对于carry和value的值都没有影响
            carry, value = divmod(value, 10)
            #利用value的值创建一个链表节点，并让current.next指向它
            current.next = ListNode(value)
            #移动current指针到下一个节点
            current = current.next
        #最后只要返回dummy的下一个节点就是我们想要的答案。
        return dummy.next
""""
二，BFS
1, 齐头并进的广度优先遍历问题：
(1) 树的宽度优先遍历
"""
# 102. Binary Tree Level Order Traversal
# Given a binary tree, return the level order traversal of its nodes' values. (ie, from left to right, level by level
class TreeNode:
    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None
from collections import deque
class Solution:
    def levelOrder(self,root:TreeNode) -> List[List[int]]:
        if not root:
            return []
        queue = deque([root])
        res =[]
        while queue:
            level =[]
            for _ in range(len(queue)):
                node = queue.popleft()
                level.append(node.val)
                if node.left:
                    queue.append(node.left)
                if node.right:
                    queue.append(node.right)
            res.append(level)
        return res
# 107. Binary Tree Level Order Traversal II
# Given a binary tree, return the bottom-up level order traversal of its nodes' values. (ie, from left to right, level by level from leaf to right
class TreeNode:
    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None
from collections import deque
class Solution:
    def levelOrderBottom(self, root:TreeNode) -> List[List[int]]:
        if not root:
            return []
        queue = deque([root])
        res = []
        while queue:
            level = []
            for _ in range(len(queue)):
                node = queue.popleft()
                level.append(node.val)
                if node.left:
                    queue.append(node.left)
                if node.right:
                    queue.append(node.right)
            res.insert(0,level)
        return res
# 103. Binary Tree Zigzag Level Order Traversal
# Given a binary tree, return the zigzag level order traversal of its nodes' values. (ie, from left to right, then right to left for the next level and alternate between)
class TreeNode:
    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None
from collections import deque
class Solution:
    def zigzagLevelOrder(self, root: TreeNode) -> List[List[int]]:
        if not root:
            return []
        queue = deque([root])
        count =0
        res = []
        while queue:
            count +=1
            level =[]
            for _ in range(len(queue)):
                node = queue.popleft()
                if count%2 ==1:
                    level.append(node.val)
                else:
                    level.insert(0,node.val)
                if node.left:
                    queue.append(node.left)
                if node.right:
                    queue.append(node.right)
            res.append(level)
        return res
"""
（2）无权图的最短路径
--依赖的数据结构：队列
--应用：无权图中找到最短路径
--注意事项：无权图中遍历，加入队列之后，必须马上标记【已经访问】
在图中，由于 图中存在环，和深度优先遍历一样，广度优先遍历也需要在遍历的时候记录已经遍历过的结点。
特别注意：将结点添加到队列以后，一定要马上标记为「已经访问」，否则相同结点会重复入队，
这一点在初学的时候很容易忽略。如果很难理解这样做的必要性，建议大家在代码中打印出队列中的元素进行调试：
在图中，如果入队的时候不马上标记为「已访问」，相同的结点会重复入队，这是不对的。另外一点还需要强调，
广度优先遍历用于求解「无权图」的最短路径，因此一定要认清「无权图」这个前提条件。
如果是带权图，就需要使用相应的专门的算法去解决它们
"""
# 323. 无向图中连通分量的数目
# 给定编号从 0 到 n-1 的 n 个节点和一个无向边列表（每条边都是一对节点），请编写一个函数来计算无向图中连通分量的数目。
"""BFS"""
class Solution:
    def countComponents(self, n: int, edges: List[List[int]]) -> int:
        from collections import defaultdict, deque
        graph = defaultdict(list)
        #只有当list里面是两个元素才可以使用这样的方式
        for x, y in edges:
            graph[x].append(y)
            graph[y].append(x)
        visited = set()
        connected = 0
        def bfs(i):
            queue = deque([i])
            while queue:
                i = queue.pop()
                for j in graph[i]:
                    if j not in visited:
                        visited.add(j)
                        queue.append(j)
        for i in range(n):
            if i not in visited:
                connected+=1
                visited.add(i)
                bfs(i)
        return connected
"""DFS"""
class Solution:
    def countComponents(self, n: int, edges: List[List[int]]) -> int:
        from collections import defaultdict, deque
        graph = defaultdict(list)
        for x, y in edges:
            graph[x].append(y)
            graph[y].append(x)
        visited = set()
        connected = 0
        def dfs(i):
            visited.add(i)
            for j in graph[i]:
                if j not in visited:
                    dfs(j)
        for i in range(n):
            if i not in visited:
                connected+=1
                dfs(i)
        return connected
"""
2, 二维平面上的搜索问题
"""
#695，岛屿的最大面积
#给定一个包含了一些0和1的费控二维数组grid,一个岛屿是由一些相邻的1（代表土地）
#构成的组合，这里的相邻要求两个1必须在水平或者竖直方向上相邻，可以假设grid的
#四个边缘都被0（代表水）包围着，找到给定的二维数组中最大的岛屿面积（如果没有返回0）
"""BFS"""
class Solution:
    def maxAreaOfIsland(self, grid: List[List[int]]) -> int:
        from collections import deque
        ans = 0
        for i, l in enumerate(grid):
            for j,n in enumerate(l):
                cur = 0
                q = deque([(i,j)])
                while q:
                    cur_i, cur_j = q.popleft()
                    # put all lands into queue and pass water
                    if cur_i<0 or cur_j<0 or cur_i == len(grid) or cur_j ==len(grid[0]) or grid[cur_i][cur_j] !=1:
                        continue
                    cur +=1
                    grid[cur_i][cur_j] = 0
                    for di, dj in [[0,1],[0,-1],[1,0],[-1,0]]:
                        next_i, next_j = cur_i+di, cur_j+dj
                        q.append((next_i,next_j))
                ans = max(ans,cur)
        return ans
"""DFS"""
# Time complex: O(R*C),R是给定的网格中的行数, C是列数，我们访问每个网格最
# Space complex: O(R*C),
class Solution:
    def maxAreaOfIsland(self, grid: List[List[int]]) -> int:
        from collections import deque
        ans = 0
        def dfs(self,grid,cur_i,cur_j):
            if cur_i < 0 or cur_j < 0 or cur_i == len(grid) or cur_j == len(grid[0]) or grid[cur_i][cur_j] != 1:
                return 0
            grid[cur_i][cur_j] = 0
            ans =1
            for di, dj in [[0,1],[0,-1],[1,0],[-1,0]]:
                next_i, next_j = cur_i + di, cur_j + dj
                ans += self.dfs(grid,next_i,next_j)
            return ans
        for i, l in enumerate(grid):
            for j,n in enumerate(l):
                ans = max(self.dfs(grid,i,j),ans)
        return ans
""""
三，排序
1, 时间复杂度 O(n^2) 级排序算法
1）冒泡排序
是入门级的算法，但也有一些有趣的玩法。通常来说，冒泡排序有三种写法：
一边比较一边向后两两交换，将最大值 / 最小值冒泡到最后一位；
经过优化的写法：使用一个变量记录当前轮次的比较是否发生过交换，如果没有发生交换表示已经有序，不再继续排序；
进一步优化的写法：除了使用变量记录当前轮次是否发生交换外，再使用一个变量记录上次发生交换的位置，下一轮排序时到达上次交换的位置就停止比较。
2, 时间复杂度 O(nlogn) 级排序算法
1）希尔排序
虽然原始的希尔排序最坏时间复杂度仍然是 O(n^2)，但经过优化的希尔排序可以达到 O(n^{1.3}) 甚至 O(n^{7/6})
希尔排序本质上是对插入排序的一种优化，它利用了插入排序的简单，又克服了插入排序每次只交换相邻两个元素的缺点。它的基本思想是：
将待排序数组按照一定的间隔分为多个子数组，每组分别进行插入排序。这里按照间隔分组指的不是取连续的一段数组，而是每跳跃一定间隔取一个值组成一组
逐渐缩小间隔进行下一轮排序
最后一轮时，取间隔为 11，也就相当于直接使用插入排序。但这时经过前面的「宏观调控」，数组已经基本有序了，所以此时的插入排序只需进行少量交换便可完成
2）堆排序
https://www.interviewcake.com/concept/java/heapsort
堆：符合以下两个条件之一的完全二叉树：
根节点的值 ≥ 子节点的值，这样的堆被称之为最大堆，或大顶堆；
根节点的值 ≤ 子节点的值，这样的堆被称之为最小堆，或小顶堆。
堆排序过程如下：
用数列构建出一个大顶堆，取出堆顶的数字；
调整剩余的数字，构建出新的大顶堆，再次取出堆顶的数字；
循环往复，完成整个排序。
整体的思路就是这么简单，我们需要解决的问题有两个：
如何用数列构建出一个大顶堆；
取出堆顶的数字后，如何将剩余的数字调整成新的大顶堆。
构建大顶堆 & 调整堆
构建大顶堆有两种方式：
方案一：从 0 开始，将每个数字依次插入堆中，一边插入，一边调整堆的结构，使其满足大顶堆的要求；
方案二：将整个数列的初始状态视作一棵完全二叉树，自底向上调整树的结构，使其满足大顶堆的要求。
第二种方案更加常用。
在介绍堆排序具体实现之前，我们先要了解完全二叉树的几个性质。将根节点的下标视为 0，则完全二叉树有如下性质：
对于完全二叉树中的第 i 个数，它的左子节点下标：left = 2i+1
对于完全二叉树中的第 i 个数，它的右子节点下标：right = left + 1
对于有 n 个元素的完全二叉树(n≥2)(n≥2)，它的最后一个非叶子结点的下标：n/2 - 1
"""
def heapSort(arr):
    # 构建初始大顶堆
    build_max_heap(arr)
    lenth = len(arr)
    for i in range(lenth-1, 0, -1):
        # 将最大值交换到数组最后
        swap(arr,0,i)
        # 调整剩余数组，使其满足大顶堆
        max_heapify(arr,0,i)
# 桂建初始大顶堆
def build_max_heap(arr):
    n = len(arr)
    for i in range(n/2-1,0,-1):
        max_heapify(arr,i,n)
# 调整大顶堆，第三个参数表示剩余未排序的数量，也是剩余堆的大小
def max_heapify(arr, i, heap_size):
    # 左子节点下标
    l = 2*i+1
    # 右子节点下标
    r = l+1
    largest = i
    # 与左子树节点比较
    if heap_size >1 and arr[l]>arr[largest]:
        largest = l
    # 与右子树节点比较
    if heap_size >1 and arr[r]>arr[largest]:
        largest = r
    if largest != i:
        # 将最大值交换为根节点
        swap(arr, i, largest)
        # 再次调整交换后的最大顶堆
        max_heapify(arr,largest,heap_size)
def swap(arr, i, j):
    temp = arr[i]
    arr[i] = arr[j]
    arr[j] = arr[i]
"""2, 时间复杂度 O(nlogn) 级排序算法
3）快排
时间复杂度O(nlogn)，时间复杂度也是O(nlogn)
基本思想：
step1: 从数组中取出一个数，称之为基数(pivot)
step2: 遍历数组，将比基数大的数字放在其右边，比基数小的放在左边，遍历完成，数组被分成左右两个区域
step3: 将两个区域视为两个数组，重复前面两个步骤，直到完成为止
"""
class Solution:
    def sortArray(self, nums: List[int]) -> List[int]:
        if not nums or len(nums) == 0:
            return
        self.quickSort(nums, 0, len(nums) - 1)
        return nums
    def quickSort(self, nums, start, end):
        if start >= end:
            return
        left, right = start, end
        pivot = (nums[left] + nums[right]) / 2
        while left <= right:
            while left <= right and nums[left] < pivot:
                left += 1
            while left <= right and nums[right] > pivot:
                right -= 1
            if left <= right:
                nums[right], nums[left] = nums[left], nums[right]
                left += 1
                right -= 1
        self.quickSort(nums, start, right)
        self.quickSort(nums, left, end)

#347, Top K Frequent Elements
class Solution:
    def findKthLargest(self, nums: List[int], k: int) -> int:
        if not nums or len(nums) == 0:
            return
        self.qucikSelect(nums, 0, len(nums) - 1, k)
        return nums[k - 1]
    def qucikSelect(self, nums, start, end, k):
        if start >= end:
            return
        left, right = start, end
        pivot = (nums[left] + nums[right]) / 2
        while left <= right:
            while left <= right and nums[left] > pivot:
                left += 1
            while left <= right and nums[right] < pivot:
                right -= 1
            if left <= right:
                nums[right], nums[left] = nums[left], nums[right]
                left += 1
                right -= 1
        if start + k - 1 <= right:
            self.qucikSelect(nums, start, right, k)
        if start + k - 1 >= left:
            self.qucikSelect(nums, left, end, k - (left - start))
""""
四，二叉树
二叉树的遍历问题或者其他的任何问题，都考虑把它落实在每个子树上，然后在每颗字数上考虑，推广到全局
1，二叉树的前/中/后序遍历问题
"""
# 144. Binary Tree Preorder Traversal。 Given a binary tree, return the preorder traversal of its nodes' values
class TreeNode:
    def __init__(self,x):
        self.val = x
        self.left = None
        self.right = None
"""非递归的方式--使用stack的方法：因为栈先进后出，可以pop右边的能力"""
class Solution:
    def preorderTraversal(self,root:TreeNode): # root,left,right
        if not root:
            return []
        stack = [root]
        res = []
        while stack:
            node = stack.pop()
            res.append(node.val)
            if node.right:
                stack.append(node.right)
            if node.left:
                stack.append(node.left)
        return res
"""递归的方式--使用递归方法"""
class Solution:
    def preorderTraversal(self,root:TreeNode): # root,left,right
        if not root:
            return []
        res = []
        self.traverse(root,res)
        return res
    def traverse(self,root,res):
        if not root:
            return
        res.append(root.val)
        self.traverse(root.left,res)
        self.traverse(root.right,res)
# 94. Binary Tree Inorder Traversal.  Given a binary tree, return the inorder traversal of its nodes' values
"""递归的方式--使用递归方法"""
class Solution:
    def inorderTraversal(self,root:TreeNode): # left,root,right
        if not root:
            return []
        res = []
        self.traverse(root,res)
        return res
    def traverse(self,root,res):
        if not root:
            return
        self.traverse(root.left,res)
        res.append(root.val)
        self.traverse(root.right,res)
"""非递归的方式--使用stack的方式"""
class Solution:
    def inorderTraversal(self,root:TreeNode): # left,root,right
        if not root:
            return []
        stack = []
        res = []
        while root:
            res.append(root)
            root=root.left
        while stack:
            node =stack.pop()
            res.append(node.val)
            if node.right:
                node1 = node.right
                while node1:
                    stack.append(node1)
                    node1 = node1.left
        return res
#144. Binary Tree Preorder Traversal.Given a binary tree, return the preorder traversal of its nodes' values
"""递归的方式--使用递归方法"""
class Solution:
    def postorderTraversal(self, root: TreeNode):# left,right,root
        if not root:
            return []
        res = []
        self.traverse(root, res)
        return res
    def traverse(self, root, res):
        if not root:
            return
        self.traverse(root.left, res)
        self.traverse(root.right, res)
        res.append(root.val)
#105. Construct Binary Tree from Preorder and Inorder Traversal
class Solution:
    def reConstructBinaryTree(self, pre, tin):
        if not pre or not tin:
            return None
        rootVal = pre[0]
        id = tin.index(pre[0])
        root = TreeNode(rootVal)
        root.left = self.reConstructBinaryTree(pre[1:id+1],tin[:id])
        root.right = self.reConstructBinaryTree(pre[id+1:],tin[id+1:])
        return root
""""
四，二叉树的基本变化序
1, DFS遍历，一般不需要返回值。但是可以在遍历的过程中进行各种操作，以达到目的
"""
# 226. Invert Binary Tree
class TreeNode:
    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None
class Solution:
    def invertTree(self,root):
        if not root:
            return None
        self.dfs(root)
        return root
    def dfs(self,root):
        if not root:
            return
        left =root.left
        right = root.right
        root.left = right
        root.right = left
        if root.right:
            self.dfs(root.right)
        if root.left:
            self.dfs(root.left)
""""
2，分治法：分治法一般要返回值，为了求最大值，需要问题转化成解决左子树上深度，
和右子树深度。然后合并中最大的就是全局深度。因为每次遍历深度加一。
"""
#101. Symmetric Tree。Given a binary tree, check whether it is
# a mirror of itself (ie, symmetric around its center。
class TreeNode:
    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None
class Solution:
    def isSymmetric(self, root: TreeNode):
        if not root:
            return True
        return self.checkSymmetric(root.left, root.right)
    def checkSymmetric(self,nodeA,nodeB):
        if nodeA is None or nodeB is None:
            return False
        if nodeA is None and nodeB is None:
            return True
        if nodeA.val != nodeB.val:
            return False
        inner_res = self.checkSymmetric(nodeA.left,nodeB.right)
        outer_res = self.checkSymmetric(nodeA.right, nodeB.left)
        return inner_res and outer_res
""""
3，Path Sum类型问题一般分为三种方式:
1) 判断是否存在从root- leaf 和为指定的数字的路径
（1）构造一个递归出口，一般就是在root 为空时候
（2）构造一个到leaf节点时候 找到指定路径的情况，
    一般就是，当root.left和root.right为空，root.val == 此时target数字
（3）然后进行左右递归，注意，参数中的，root和target数字都是要需要更新的
"""
#112. Path Sum. Given a binary tree and a sum, determine if the tree has a root-to-leaf path such that
# adding up all the values along the path equals the given sum.
class TreeNode:
    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None
class Solution:
    def hasPathSum(self,root,sum):
        if root is None:
            return False
        if root.left is None and root.left is None:
            return sum == root.val
        return self.hasPathSum(root.left, sum-root.val) or self.hasPathSum(root.right, sum-root.val)

#437. Path Sum III. You are given a binary tree in which each node contains an integer value
#Find the number of paths that sum to a given value.The path does not need to start or end at
#the root or a leaf, but it must go downwards (traveling only from parent nodes to child nodes)
#The tree has no more than 1,000 nodes and the values are in the range -1,000,000 to 1,000,000
class TreeNode:
    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None
class Solution:
    def pathSum(self, root: TreeNode, sum: int):
 # 思路分析： 因为题目中要求，只要path之和等于所给的sum，可以不必须是从root开始，而是可以从任何地方开始。
 # 那么就需要考虑，root, root.left, root.right 三种情况下的。总的path的数量就是他们三个所包含的之和
        if not root:
            return 0
        res = []
        self.helper(root, [], sum, res)
        return res
    def helper(self, root, path, target, res):
        if not root:
            return
        path += [root.val]
        flag = (root.left == None and root.right === None)
        if root.val ==target and flag:
            res.append(path[:])
            return
        if root.left:
            self.helper(root.left, path[:], target-root.val,res)
        if root.right:
            self.helper(root.right, path[:], target-root.val,res)
        path.pop()
"""
4，关于最大最小长度问题：一般采用的是分治法
(1) 一般需要返回三类值:1)curLen/curSum, 返回当前所求的长度和值；2）maxLen/maxSum，返回最值; 3) node或者nodeVal,
返回节点或者节点数值，具体选择哪一个，看题中要求；
(2) 对于空集的判断if not root,需要注意的是，最值在初始化的时候，最大值一般是用-sys.maxsizem,最小值是用sys.maxsize
(3) divide，分别对左右进行循环
(4) conque, 在conque的时候，需要分两步:1)先根据调解，求出curLen/curSum；2）然后在判断出最值
"""
# Minimum Subtree.Given a binary tree, find the subtree with minimum sum. Return the root of the subtree
class TreeNode:
    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None
class Solution:
    def findSubtree(self, root: TreeNode):
        if not root:
            return None
        node, minSubtree,Sum = self.helper(root)
        return node
    def helper(self,root):
        if not root:
            return None, 0, sys.maxsize
        left_node, left_min_sum, left_sum = self.helper(root.left)
        right_node, right_min_sum, right_sum = self.helper(root.right)
        total_sum = left_sum + root.val+ right_sum
        if left_min_sum == min(left_min_sum, right_min_sum, total_sum):
            return left_node, left_min_sum, total_sum
        if right_min_sum == min(left_min_sum, right_min_sum, total_sum):
            return right_node , right_min_sum, total_sum
        if total_sum == min(left_min_sum, right_min_sum, total_sum):
            return root, total_sum, total_sum
# 298. Binary Tree Longest Consecutive Sequence.Given a binary tree, find the length of the longest consecutive sequence path
# The path refers to any sequence of nodes from some starting node to any node in the tree along the parent-child connections.
# The longest consecutive path need to be from parent to child (cannot be the reverse)
class TreeNode:
    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None
class Solution:
    def longestConsecutive(self, root: TreeNode):
        if not root:
            return 0
        maxLen, curLen, curVal = self.helper(root)
        return maxLen
    def helper(self, root):
        if not root:
            return -sys.maxsize, 0, 0
        left_maxLen, left_curLen, left_curVal = self.helper(root.left)
        right_maxLen, right_curLen, right_curVal = self.helper(root.right)
        curLen =1
        if root.val == left_curVal -1:
            curLen = max(curLen, left_curLen+1)
        if root.val == right_curVal -1:
            curLen = max(curLen, right_curLen+1)
        if left_maxLen == max(curLen, left_maxLen, right_maxLen):
            return left_maxLen, curLen, root.val
        if right_maxLen == max(curLen, left_maxLen, right_maxLen):
            return right_maxLen, curLen, root.val
        return curLen,curLen,root.val
"""
5, 二叉树宽度有限搜索BFS：主要是使用deque这类函数先进先出的优点，配合deque.popleft(), 实现层序遍历
"""
# 102. Binary Tree Level Order Traversal。Given a binary tree, return the level order traversal
# of its nodes' values. (ie, from left to right, level by level)
class TreeNode:
    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None
class Solution:
    def levelOrder(self, root: TreeNode):
        if not root:
            return []
        queue = deque([root])
        res = []
        while queue:
            level = []
            for _ in range(len(queue)):
                node = queue.popleft()
                level.append(node.val)
                if node.left:
                    queue.append(node.left)
                if node.right:
                    queue.append(node.right)
            res.append(level)
        return res
# 107, Binary Tree Level Order Traversal II.Given a binary tree, return the bottom-up level order traversal of its nodes' values.
# (ie, from left to right, level by level from leaf to root)
class TreeNode:
    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None
from collections import deque
class Solution:
    def levelOrderBottom(self, root: TreeNode):
        if not root:
            return []
        queue = deque([root])
        res = []
        while queue:
            level = []
            for _ in range(len(queue)):
                node = queue.popleft()
                level.append(node.val)
                if node.left:
                    queue.append(node.left)
                if node.right:
                    queue.append(node.right)
            res.insert(0,level)
        return res
#103,Binary Tree Zigzag Level Order Traversal.
#Given a binary tree, return the zigzag level order traversal of its nodes' values. (ie, from left to right,
# then right to left for the next level and alternate between)."
class TreeNode:
    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None
from collections import deque
class Solution:
    def zigzagLevelOrder(self, root: TreeNode) -> List[List[int]]:
        if not root:
            return []
        queue = deque([root])
        res = []
        while queue:
            count +=1
            level = []
            for _ in range(len(queue)):
                node = queue.popleft()
                if count%2 ==1:
                    level.append(node.val)
                else:
                    level.insert(0,node.val)
                if node.left:
                    queue.append(node.left)
                if node.right:
                    queue.append(node.right)
            res.append(level)
        return res
"""
6,二叉搜索树(binary search tree)相关问题.二叉搜索树的条件：1）左子树值小于根节点； 2）根节点值小于右子树； 3）左右子树都是二叉搜索树BST
"""
#426. Convert Binary Search Tree to Sorted Doubly Linked List.Convert a BST to a sorted circular doubly-linked list in-place.
# Think of the left and right pointers as synonymous to the previous and next pointers in a doubly-linked list.For a circular
# doubly linked list, the predecessor of the first element is the last element, and the successor of the last element is the first element.
class TreeNode:
    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None
from collections import deque
class Solution:
    def treeToDoublyList(self, root: TreeNode) -> List[List[int]]:
        if not root:
            return []
        res = []
        array = self.inOrderTraversal(root,res)
        head = self.constructDoublyLinkedList(array)
        return head
    def inOrderTraversal(self,root,res): # left root right
        if not root:
            return []
        res = []
        stack =[]
        while root:
            stack.append(root)
            root = root.left
        while stack:
            node = stack.pop()
            res.append(node)
            if node.right:
                node = node.right
                while node:
                    stack.append(node)
                    node = node.left
        return res
    def constructDoublyLinkedList(self, array):
        array[-1].right = array[0]
        array[0].left = array[-1]
        for i in range(len(array)):
            array[i-1].right = array[i]
            array[i].left = array[i-1]
        return array[0]
# 285. Inorder Successor in BST. Given a binary search tree and a node in it, find the in-order successor of that node in the BST
#The successor of a node p is the node with the smallest key greater than p.val
# 中序后继: 二叉树中序是 left root right, 二叉搜索树，左子树小于根节点，根节点小于右子树，左右子树都是二叉树。
# 二叉搜索树的中序遍历是个递增的数组，顺序后继是中序遍历中当前节点 之后最小的节点
# 可以分为两种情况来讨论：1）如果当前节点有右孩子，顺序后继在当前节点之下；如果当前节点没有右孩子，顺序后继在当前节点之上。
# 算法逻辑： 1）如果当前节点有右子树，找到右子树，再持续往左直到节点左孩子为空，直接返回该节点；2）如果当前节点没有右子树，就需要使用非递归
# 的中序遍历，维持一个栈，当栈中有节点时：a,往左走直到节点的左孩子为空，并将每个访问的节点压入栈中；b,弹出栈中节点，判断当前的前继节点是否为p
# 如果是则直接返回当前节点，如果不是，将当前节点赋值给前继节点；c, 往右走一步；3）如果走到这一步，说明不存在后续，返回空
class Solution:
    def inorderSuccessor(self, root: 'TreeNode', p: 'TreeNode') -> 'TreeNode':
        if p.right:
            p = p.right
            while p.left:
                p = p.left
            return p
        stack = []
        inorder_precessor_val = float('-inf') # set a value as the precessor's
        # if p have no right node, the inordersuccessor must be somewhere as the right node of it's smallest left nodes
        while root or stack:
            # go to left till the smallest left node
            while root:
                stack.append(root)
                root = root.left
            root = stack.pop()  # the smallest left node
            if inorder_precessor_val == p.val:
                return root
            inorder_precessor_val = root.val
            root = root.right
        return None
# 时间复杂度： 如果节点p有右孩子，时间复杂度为O(H_p）其中H_p是节点p的高度。如果没有右孩子，时间复杂度为O(H)，其中H为树的高度
# 空间复杂度： 如果节点p有右孩子，空间复杂度为O(1).如果没有右孩子，空间复杂度为O(H)
#1008.Construct Binary Search Tree from Preorder Traversal. Given an array of integers
class Solution:
    def bstFromPreorder(self, preorder: List[int]) -> TreeNode:
        root_val = preorder[0]
        root = TreeNode(root_val)
        stack = []
        stack.append(root)
        for value in preorder[1:]:
            if stack[-1].val>value:
                stack[-1].left = TreeNode(value)
                stack.append(stack[-1].left)
            else:
                while stack and stack[-1].val<value:
                    last = stack.pop()
                last.right = TreeNode(value)
                stack.append(last.right)
        return root
# 230. Kth Smallest Element in a BST. Given a binary search tree, write a function kthSamallest to find the
# kth smallest element in it. You may assume k is always valid, 1 ≤ k ≤ BST's total elements.
# 搜索二叉树的Inorder Traversal 所得到的就是递增数列
class Solution:
    def kthSmallest(self, root: TreeNode, k: int) -> int:
        if not root:
            return None
        in_order_traversal = self.inOrderTraveral(root)
        return in_order_traversal[k - 1]
    def inOrderTraveral(self, root):  # left root right
        if not root:
            return []
        res = []
        stack = []
        while root:
            stack.append(root)
            root = root.left
        while stack:
            node = stack.pop()
            res.append(node.val)
            if node.right:
                node1 = node.right
                while node1:
                    stack.append(node1)
                    node1 = node1.left
        return res
# 236. Lowest Common Ancestor of a Binary Tree.Given a binary tree, find the lowest common ancestor (LCA)
# of two given nodes in the tree. According to the definition of LCA on Wikipedia: “The lowest common
# ancestor is defined between two nodes p and q as the lowest node in T that has both p and q as descendants
# (where we allow a node to be a descendant of itself)
class Solution(object):
    def lowestCommonAncestor(self, root, p, q):
        if not root:
            return None
        if root == p or root == q:
            return root
        left_result = self.lowestCommonAncestor(root.left, p, q)
        right_result = self.lowestCommonAncestor(root.right, p, q)
        if left_result and right_result:
            return root
        if left_result:
            return left_result
        if right_result:
            return right_result
        return None
""""
TX的top 10算法
1， CSIG： 
补充题4. 手撕快速排序；206. 反转链表；704. 二分查找； 415. 字符串相加； 补充题6. 手撕堆排序；102. 二叉树的层序遍历；
470. 用 Rand7() 实现 Rand10()； 53. 最大子序和； 4. 寻找两个正序数组的中位数； 141. 环形链表	
2，IEG：
146. LRU缓存机制；160. 相交链表； 1. 两数之和；155. 最小栈；
232. 用栈实现队列；21. 合并两个有序链表；25. K 个一组翻转链表；
3，PCG：
215. 数组中的第K个最大元素；
5. 最长回文子串；70. 爬楼梯；15. 三数之和；补充题23. 检测循环依赖；
4，TEG：
121. 买卖股票的最佳时机；
补充题22. IP地址与整数的转换；236. 二叉树的最近公共祖先；
5，CDG：
；227. 基本计算器 II；143. 重排链表；20. 有效的括号；144. 二叉树的前序遍历；460. LFU缓存； 
3. 无重复字符的最长子串；118. 杨辉三角； 31. 下一个排列；
6，WXG：
8. 字符串转换整数 (atoi)；153. 寻找旋转排序数组中的最小值；
剑指 Offer 54. 二叉搜索树的第k大节点； 300. 最长上升子序列；2. 两数相加；112. 路径总和
"""
# 补充题4. 手撕快速排序
class Solution:
    def sortArray(self, nums: List[int]) -> List[int]:
        if not nums:
            return []
        self.quickSort(nums, 0, len(nums)-1)
        return nums
    def quickSort(self,nums,start, end):
        if start >=end:
            return
        left,right = start, end
        pivot = (nums[left]+nums[right])/2
        while left <= right :
            while left <= right and nums[left]<pivot:
                left+=1
            while left>=right and nums[right]>pivot:
                right-=1
            if left<=right:
                nums[left], nuums[right] = nums[right], nums[left]
                left+=1
                right-=1
        self.quickSort(nums, start, right)
        self.quickSort(nums, left, end)
# 206. 反转链表
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution:
    def reverseList(self, head: ListNode) -> ListNode:
        cur = None
        while head!=None:
            tmp = head.next
            head.next = cur
            cur = head
            head = tmp
        return cur
# 704. 二分查找
#Given an array of integers nums which is sorted in ascending order, and an integer target, write a function to search target in nums.
#If target exists, then return its index. Otherwise, return -1.You must write an algorithm with O(log n) runtime complexity.
class Solution:
    def search(self, nums: List[int], target: int) -> int:
        if not nums or target is None:
            return -1
        start, end = 0, len(nums)-1
        while start<=end:
            mid = (start+end)//2
            if nums[mid] == target:
                return mid
            elif nums[mid] > target:
                end = mid -1
            else:
                start = mid+1
        return -1
#415. 字符串相加，
#Given two non-negative integers, num1 and num2 represented as string, return the sum of num1 and num2 as a string.
#You must solve the problem without using any built-in library for handling large integers (such as BigInteger).
#You must also not convert the inputs to integers directly.
class Solution:
    def addStrings(self, num1: str, num2: str) -> str:
        res = ""
        i,j,carry = len(num1)-1, len(num2)-1,0
        while i>=0 or j>=0:
            n1 =int(num1[i]) if i>=0 else 0
            n2 = int(num2[j]) if j>=0 else 0
            tmp = n1+n2+carry
            carry = tmp//10
            res = str(tmp%10)+res
            i,j=i-1,j-1
        return "1"+res if carry else res
#补充题6. 手撕堆排序
#堆就是一个完全二叉树，完全二叉树的定义：深度为h的完全二叉树，除了第h层外，其他各层(1~h-1)的节点数都达到最大，第h层所有的节点都集中在最左边。
#如果堆总共有n个节点，那么堆的最后一个非叶子节点是第n//2-1
def heapSort(arr):
    build_max_heap(arr)
    lenth = len(arr)
    for i in range(lenth-1, 0, -1):
        swap(arr,0, i)
        max_heapify(arr,0,i)
def build_max_heap(arr):
    n = len(arr)
    for i in range(n/2-1, 0, -1):
        max_heapify(arr,i,n)
def max_heapify(arr,i,heap_size):
    l = 2*i+1
    r = l+1
    largest = i
    if heap_size>1 and arr[l]>arr[largest]:
        largest = l
    if heap_size>1 and arr[r]>arr[largest]:
        largest = l
    if largest != i:
        swap(arr,i,largest)
        max_heapify(arr, largest, heap_size)
def swap(arr, i, j):
    temp = arr[i]
    arr[i] = arr[j]
    arr[j] = arr[i]
# 102. 二叉树的层序遍历
# 102. Binary Tree Level Order Traversal
# Given a binary tree, return the level order traversal of its nodes' values. (ie, from left to right, level by level
class TreeNode:
    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None
from collections import deque
class Solution:
    def levelOrder(self,root:TreeNode) -> List[List[int]]:
        if not root:
            return []
        queue = deque([root])
        res = []
        while queue:
            level = []
            for _ in range(len(queue)):
                node = queue.popleft()
                level.append(node.val)
                if node.left:
                    queue.append(node.left)
                if node.right:
                    queue.append(node.right)
                res.append(level)
        return res
# 470. 用 Rand7() 实现 Rand10()
# Given the API rand7() that generates a uniform random integer in the range [1, 7], write a function rand10() that generates a uniform random integer in the range [1, 10].
# You can only call the API rand7(), and you shouldn't call any other API. Please do not use a language's built-in random API. Each test case will have one internal argument n,
# the number of times that your implemented function rand10() will be called while testing. Note that this is not an argument passed to rand10().
class Solution:
    def rand10(self):
        # 先在0-5之间均匀分布，去掉6，7，然后再加一个0或者5，就可以映射到0-10之间
        n = rand7()
        while n >5: #使用while的目的，是满足均匀概率分布
            n = rand7()
        i = rand7()
        while i ==4:
            i = rand7()
        if i > 5:
            j = 0
        else:
            j = 5
        return n+j
# 53. 最大子序和
#Given an integer array nums, find the contiguous subarray (containing at least one number) which has the largest sum and return its sum.
# A subarray is a contiguous part of an array.
class Solution:
    def maxSubArray(self,nums):
        if not nums or len(nums)==0:
            return 0
        n = len(nums)
        f = n*[0]
        f[0]=nums[0]
        for i in range(1,n):
            f[i] = nums[i]+max(f[i-1],0)
        max_value = -sys.maxsize -1
        for v in f:
            max_value = max(v, max_value)
        return max_value
# 4. 寻找两个正序数组的中位数
# Given two sorted arrays nums1 and nums2 of size m and n respectively, return the median of the two sorted arrays.
# The overall run time complexity should be O(log (m+n)).
# 总体思路：要求得到的中位数，如果两个数组直接合并到一起，使用merge_sort的方式，复杂度是O(m+n)，不符合要求，但是可以合并到一起分析出来，如果合并的数组
# 是偶数，得到的结果就是中间两个数的中位数；如果是奇数，就是中间的那个数。
# 因为是log的复杂度，所以一定是二分法，因为两个序列都是sorted的，那么可以把两个数组都拆分成左右两个部分，中间的那个数便是要找的数。
# nums1被拆开为L1,R1；nums2被拆开为L2,R2。那么最终找到的中位数，其左边的数一定是L1+L2（为两个数列数量之和的一半）；然后还需要满足的条件：
# L1<R1&&L1<R2 L2<R1&&L2<R2
class Solution:
    def findMedianSortedArrays(self, nums1: List[int], nums2: List[int]) -> float:
        # choose the nums with smaller length to implement sort as the time complexity would be O(log(min(m,n)))
        n1, n2 = len(nums1), len(nums2)
        total_len = n1 + n2
        if n1 > n2:
            return self.findMedianSortedArrays(nums2, nums1)
        if n1 == 0:
            if n2 % 2!= 0:
                return nums2[total_len // 2]
            else:
                return nums2[total_len // 2 - 1] + nums2[total_len // 2]
        left_edge, right_edge = 0, n1
        # cur1, cur2 represent the position in nums1, nums2
        result = 0
        while left_edge <= right_edge:
            cur1 = (left_edge + right_edge) // 2
            cur2 = (total_len + 1) // 2 - cur1
            # figure out the L1,R1,L2,R2
            if cur1 == 0:
                L1 = -sys.maxsize
                R1 = nums1[cur1]
            elif cur1 == n1:
                L1 = nums1[cur1 - 1]
                R1 = sys.maxsize
            else:
                L1 = nums1[cur1 - 1]
                R1 = nums1[cur1]
            if cur2 == 0:
                L2 = -sys.maxsize
                R2 = nums2[cur2]
            elif cur2 == n2:
                L2 = nums2[cur2 - 1]
                R2 = sys.maxsize
            else:
                L2 = nums2[cur2 - 1]
                R2 = nums2[cur2]
            # Binary search, find the boundary
            if L1 > R2:
                right_edge = cur1 - 1
            elif L2 > R1:
                left_edge = cur1 + 1
            else:
                # if the length is odd,choose the middle one. if even, choose the half of two middles' sum
                if total_len % 2 != 0:
                    result = max(L1, L2)
                else:
                    result = (max(L1, L2) + min(R1, R2)) / 2
                break
        return result
# 141. 环形链表 Given a linked list, determine if it has a cycle in it
class ListNode:
    def __init__(self, x):
        self.val = x
        self.next = None
class Solution:
    def hasCycle(self, head: ListNode) -> bool:
        if not head or not head.next:
            return False
        p1 = head
        p2 = head
        while p2 and p2.next:
            p1 = p1.next
            p2 = p2.next.next
            if p1 is p2:
                return True
        return False
# 146. LRU缓存机制；Design a data structure that follows the constraints of a Least Recently Used (LRU) cache.
# Implement the LRUCache class:
# LRUCache(int capacity) Initialize the LRU cache with positive size capacity.
# int get(int key) Return the value of the key if the key exists, otherwise return -1.
# void put(int key, int value) Update the value of the key if the key exists. Otherwise, add the key-value pair to the cache.
# If the number of keys exceeds the capacity from this operation, evict the least recently used key.
# The functions get and put must each run in O(1) average time complexity.
# 时间复杂度： 对于put和get都是O(1)
# 空间复杂度：O(capacity),因为哈希表和双向链表最多可以存储capacity+1个元素
class DLinkedNode:
    def __init__(self, key=0, value=0):
        self.key = key
        self.value = value
        self.prev = None
        self.next = None
class LRUCache:
    def __init__(self, capacity:int):
        self.cache = dict()
        # use dummy head and dummy tail
        self.head = DLinkedNode()
        self.tail = DLinkedNode()
        self.head.next = self.tail
        self.tail.prev = self.head
        self.capacity = capacity
        self.size = 0
    def get(self, key:int):
        if key not in self.cache:
            return -1
        # if key exist, locating the key by hash and then move to the head of dlinked'
        node = self.cache[key]
        self.moverToHead(node)
        retuurn node.value
    def put(self, key, value):
        if key not in self.cache:
            # if key not exist, create a new node
            node = DLinkedNode(key, value)
            # add to hash
            self.cache[key] = node
            # add to the head of dlinked list
            self.addToHead(node)
            self.size+=1
            if self.size > self.capacity:
                # if larger than capacity, delete the tail node of dlinked
                removed = self.removeTail()
                self.cache.pop(removed.key)
                self.size -=1
        else:
            # if key exist, use hash to locate,then modify the value and move to the head
            node = self.cache[key]
            node.value = value
            self.moveToHead(node)
    def addToHead(self,node):
        node.prev = self.head
        node.next = self.head.next
        # remind this is double linked list
        self.head.next.prev = node
        self.head.next = node
    def removeNode(self,node):
        node.prev.next = node.next
        node.next.prev = node.prev
    def moveToHead(self, node):
        self.removeNode(node)
        self.addToHead(node)
    def removeTail(self):
        node = self.tail.prev
        self.removeNode(node)
        return node
#160. 相交链表：Given the heads of two singly linked-lists headA and headB, return the node at which the two lists intersect.
# If the two linked lists have no intersection at all, return null.
# Total length of Linkedlist A,B is a,b. The common length of intersection is c. Assign two pointers pA and pB move at the same times.
# Only a = b, can A meet B at the first round. In the second round, assign pA to head B, so the distance for A before it to the intersection
# would be a+b-c. assign pB to headA, so the distance for B before it would be b+a-c. The distance of A equal B
class ListNode:
    def __init__(self, x):
        self.val = x
        self.next = None
class Solution:
    def getIntersectionNode(self, headA: ListNode, headB: ListNode) -> ListNode:
        if not headA or not headB:
            return None
        pA, pB = headA, headB
        while pA!=pB:
            if pA:
                pA = pA.next
            else:
                pA = headB
            if pB:
                pB = pB.next
            else:
                pB = headA
        return pA
# 1. 两数之和。Given an array of integers nums and an integer target, return indices of the two numbers such that they add up to target.
# You may assume that each input would have exactly one solution, and you may not use the same element twice.
# You can return the answer in any order.
class Solution:
    # use hashmap, time complexity O(n), space complexity O(n)
    def twoSum(self, nums: List[int], target: int) -> List[int]:
        if not nums:
            return []
        hashmap = {}
        for i in range(0, len(nums)):
            if target - nums[i] in hashmap:
                return [hashmap[target - nums[i]], i]
            hashmap[nums[i]] = i
        return -1
    # use two pointer, time complexity O(nlogn): sorted time complexity is O(nlogn), two pointer find target is O(n), space complexity O(n)
    def twoSum(self, nums: List[int], target: int) -> List[int]:
        if not nums:
            return []
        tmp = sorted(nums)
        left,right = 0, len(nums)-1
        while left<right:
            if tmp[left]+tmp[right] == target:
                i, j = nums.index(tmp[left]),nums.index(tmp[right])
                if i==j:
                    index = nums[i+1:].index(tmp[right])+i+1
                    return [i, index]
                elif i>j:
                    return [j, i]
                else:
                    return [i, j]
            elif tmp[left]+tmp[right] >target:
                right-=1
            else:
                left+=1
        return -1
# 155. 最小栈
# Design a stack that supports push, pop, top, and retrieving the minimum element in constant time.
# Implement the MinStack class:
# MinStack() initializes the stack object.
# void push(val) pushes the element val onto the stack.
# void pop() removes the element on the top of the stack.
# int top() gets the top element of the stack.
# int getMin() retrieves the minimum element in the stack.
# Time complexity: O(1)，all operations remove/read/push are O(1); Space complexity:O(n)
class MinStack:
    def __init__(self):
        self.stack = []
        self.min_stack = [math.inf]
    def push(self, x):
        self.stack.append(x)
        self.min_stack.append(min(x,self.min_stack[-1]))
    def pop(self):
        self.stack.pop()
        self.min_stack.pop()
    def top(self):
        self.stack[-1]
    def getMin(self):
        return self.min_stack[-1]
# 232. 用栈实现队列
    def __init__(self):
        """
        Initialize your data structure here.
        """
        self.s1 = []
        self.s2 = []
        self.front = None
    def push(self, x: int) -> None:
        """
        Push element x to the back of queue.
        """
        if not self.s1: self.front = x
        self.s1.append(x)
    def pop(self) -> int:
        """
        Removes the element from in front of queue and returns that element.
        """
        if not self.s2:
            while self.s1:
                self.s2.append(self.s1.pop())
            self.front = None
        return self.s2.pop()
    def peek(self) -> int:
        """
        Get the front element.
        """
        if self.s2:
            return self.s2[-1]
        return self.front
    def empty(self) -> bool:
        """
        Returns whether the queue is empty.
        """
        if not self.s1 and not self.s2:
            return True
        return False
# 21. 合并两个有序链表
"""一,双指针
时间复杂度一般是O(n)
Slicing Windows and two points
什么是滑动窗口？
其实就是一个队列,比如例题中的 abcabcbb，进入这个队列（窗口）为 abc 满足题目要求，当再进入 a，队列变成了 abca，这时候不满足要求。
所以，我们要移动这个队列！
如何移动？
我们只要把队列的左边的元素移出就行了，直到满足题目要求！
一直维持这样的队列，找出队列出现最长的长度时候，求出解！
时间复杂度：O(n)
1) 经典双指针使用统计非重复数量
"""
#3. Longest Substring Without Repeating Characters，Given a string, find the length of the
#longest substring without repeating characters
class Solution:
    def lengthOfLongestSubstring(self, s):
        if not s:
            return 0
        n=len(s)
        left, right = 0, 0
        max_len = 0
        visited = set([])
        for left in range(n):
            while right <n and s[right] not in visited:
                visited.add(s[right])
                right+=1
            max_len=max(max_len,right-left)
            visited.discard(s[left])
        return max_len
"""
2) 同一个数组，要in-place的去掉里面的特殊数，类似0，或者重复
Step1: 设置双指针，同一起点，注意如果有特殊需求，类似不可出现两次这样的，起点设置不一定是从0开始
Step2: 让右指针作为大循环，然后在里面先判断不满足条件的情况，让左右指针的值互换，然后左指针前行，为小循环
Step3: 返回值，注意返回值是否加上1
"""
#283. Move Zeroes，Given an array nums, write a function to move all 0's to the end of it
# while maintaining the relative order of the non-zero elements.
class Solution:
    def moveZeros(self, nums):
        left, right = 0, 0
        n = len(nums)
        while right < n:
            # right和left同一起点，如果是非零元素，他们都会同时走，有left+=1，也有right+=1，只有当出现0元素，right会先走一步，跳过0，left进入0
            if nums[right] !=0:
                if left!=right:
                    nums[left] = nums[right]
                left +=1
            right +=1
        while left < n:
            if nums[left] !=0:
               nums[left] = 0
            left +=1
        return nums
# 604, Window Sum. Given an array of n integers, and a moving window(size k), move the window at each",
# iteration from the start of the array, find the sum of the element inside the window at",
# each moving
class Solution:
    def winSum(self,nums,k):
        if not nums or k <0:
            return []
        n = len(nums)
        left, right = 0, k-1
        res = []
        while right < n:
            if left == 0:
                sum_each = sum(nums[0:right+1])
                res.append(sum_each)
            else:
                sum_each = res[-1] - nums[left-1]+nums[right]
                res.append(sum_each)
            left+=1
            right+=1
        return res
#287. Find the Duplicate Number。 Given an array nums containing n + 1 integers where each integer is
# between 1 and n (inclusive), prove that at least one duplicate number must exist. Assume that there
# is only one duplicate number, find the duplicate one
"""使用类似环形链表142题的方式，
关键是理解如何把输入的数组看作为链表。首先要明确前提，整个数组中的nums是在[1,n]之间，考虑两种情况：
1) 如果数组中没有重复的数字，以数组[1,3.4.2]为例，我们将数组下标n和数nums[n]建立映射关系f(n), 其隐射关系n->f(n)为：
0->1 
1->3
2->4
3->2
我们以下标0出发，根据f(n)计算出一个值，以这个值为新的下标，再用这个函数计算，以此类推，直到下标超界。这样就可以产生一个类似链表的
序列0->1->3->2->4->null
2) 如果数组中有重复的数字，以数组[1,3.4.2，2]为例，我们将数组下标n和数nums[n]建立映射关系f(n), 其隐射关系n->f(n)为：
0->1 
1->3
2->4
3->2
4->2
同样可以构造链表：0->1->3->2->4->3->2->4->2，从理论上讲，数组中如果有重复的数，那么就会产生多对一的映射，这样，形成的链表就一定会有环路了
综述，如果数组中有重复数字，那么链表中就存在环，找到环的入口就是找到这个数字
142中慢指针走一步 slow=slow.next 等于本题中slow=nums[slow]
142中快指针走一步 fast=fast.next 等于本题中fast=nums[nums[fast]]
"""
class Solutions:
    def findDulicate(self,nums):
        if len(nums)<=1:
            return -1
        slow = nums[0]
        fast = nums[nums[0]]
        while slow!=fast:
            slow = nums[slow]
            fast= nums[nums[fast]]
        fast = 0
        while fast!=slow:
            fast = nums[fast]
            slow = nums[slow]
        return slow
# 142. Linked List Cycle II.Given a linked list, return the node where the cycle begins.
# If there is no cycle, return null.To represent a cycle in the given linked list, we use
# an integer pos which represents the position (0-indexed) in the linked list where tail
# connects to. If pos is -1, then there is no cycle in the linked list. Note: Do not modify the linked list
class Solutions:
    def detectCycle(self, head):
        if not head or head.next:
            return -1
        slow, fast = head, head
        while fast and fast.next:
            slow = slow.next
            fast = fast.next.next
            if slow is fast:
                break
        if fast is slow:
            slow = head
            while fast is not slow:
                fast = fast.next
                slow = slow.next
            return slow
        return None
# 2. Add Two Numbers. You are given two non-empty linked lists representing two non-negative integers.
# The digits are stored in reverse order and each of their nodes contain a single digit. Add the two numbers and return it as a linked list.",
class Solutions:
    def addTwoNumbers(self,l1:ListNode,l2:ListNode):
        #首先创建一个虚拟节点，并创建一个current指针，指向这个节点
        current = dummy = ListNode()
        #初始化carry和两个链表对应节点相加的值
        carry, value = 0, 0
        #下面的while循环中之所以有carry，是为了处理两个链表最后节点相加出现进位的情况
        #当两个节点都走完而且最后的运算并没有进位时，就不会进入这个循环
        while carry or l1 or l2:
            #让value先等于carry既有利于下面两个if语句中两个对应节点值相加，
            # 也是为了要处理两个链表最后节点相加出现进位的情况
            value = carry
            #只要其中一个链表没走完，就需要计算value的值
            #如果其中一个链表走完，那么下面的计算就是加总carry和其中一个节点的值
            #如果两个链表都没走完，那么下面的计算就是carry+对应的两个节点的值
            if l1: l1, value = l1.next, l1.val + value
            if l2: l2, value = l2.next, l2.val + value
            #为了防止value值大于十，出现进位，需要特殊处理
            #如果value小于十，下面这行的操作对于carry和value的值都没有影响
            carry, value = divmod(value, 10)
            #利用value的值创建一个链表节点，并让current.next指向它
            current.next = ListNode(value)
            #移动current指针到下一个节点
            current = current.next
        #最后只要返回dummy的下一个节点就是我们想要的答案。
        return dummy.next
# 剑指 Offer 59 - I. 滑动窗口的最大值: 给定一个数组 nums 和滑动窗口的大小 k，请找出所有滑动窗口里的最大值。
# 基本思想：
# 维护一个长度小于等于k的单调递减的单调队列，不断向右滑动，该单调队列的第一个元素即为该窗口的最大元素
#
class Solution:
    def maxInWindows(self, num, size):
""""
二，BFS
1, 齐头并进的广度优先遍历问题：
(1) 树的宽度优先遍历
"""
# 102. Binary Tree Level Order Traversal
# Given a binary tree, return the level order traversal of its nodes' values. (ie, from left to right, level by level
class TreeNode:
    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None
from collections import deque
class Solution:
    def levelOrder(self,root:TreeNode) -> List[List[int]]:
        if not root:
            return []
        queue = deque([root])
        res =[]
        while queue:
            level =[]
            for _ in range(len(queue)):
                node = queue.popleft()
                level.append(node.val)
                if node.left:
                    queue.append(node.left)
                if node.right:
                    queue.append(node.right)
            res.append(level)
        return res
# 107. Binary Tree Level Order Traversal II
# Given a binary tree, return the bottom-up level order traversal of its nodes' values. (ie, from left to right, level by level from leaf to right
class TreeNode:
    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None
from collections import deque
class Solution:
    def levelOrderBottom(self, root:TreeNode) -> List[List[int]]:
        if not root:
            return []
        queue = deque([root])
        res = []
        while queue:
            level = []
            for _ in range(len(queue)):
                node = queue.popleft()
                level.append(node.val)
                if node.left:
                    queue.append(node.left)
                if node.right:
                    queue.append(node.right)
            res.insert(0,level)
        return res
# 103. Binary Tree Zigzag Level Order Traversal
# Given a binary tree, return the zigzag level order traversal of its nodes' values. (ie, from left to right, then right to left for the next level and alternate between)
class TreeNode:
    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None
from collections import deque
class Solution:
    def zigzagLevelOrder(self, root: TreeNode) -> List[List[int]]:
        if not root:
            return []
        queue = deque([root])
        count =0
        res = []
        while queue:
            count +=1
            level =[]
            for _ in range(len(queue)):
                node = queue.popleft()
                if count%2 ==1:
                    level.append(node.val)
                else:
                    level.insert(0,node.val)
                if node.left:
                    queue.append(node.left)
                if node.right:
                    queue.append(node.right)
            res.append(level)
        return res
"""
（2）无权图的最短路径
--依赖的数据结构：队列
--应用：无权图中找到最短路径
--注意事项：无权图中遍历，加入队列之后，必须马上标记【已经访问】
在图中，由于 图中存在环，和深度优先遍历一样，广度优先遍历也需要在遍历的时候记录已经遍历过的结点。
特别注意：将结点添加到队列以后，一定要马上标记为「已经访问」，否则相同结点会重复入队，
这一点在初学的时候很容易忽略。如果很难理解这样做的必要性，建议大家在代码中打印出队列中的元素进行调试：
在图中，如果入队的时候不马上标记为「已访问」，相同的结点会重复入队，这是不对的。另外一点还需要强调，
广度优先遍历用于求解「无权图」的最短路径，因此一定要认清「无权图」这个前提条件。
如果是带权图，就需要使用相应的专门的算法去解决它们
"""
# 323. 无向图中连通分量的数目
# 给定编号从 0 到 n-1 的 n 个节点和一个无向边列表（每条边都是一对节点），请编写一个函数来计算无向图中连通分量的数目。
"""BFS"""
class Solution:
    def countComponents(self, n: int, edges: List[List[int]]) -> int:
        from collections import defaultdict, deque
        graph = defaultdict(list)
        #只有当list里面是两个元素才可以使用这样的方式
        for x, y in edges:
            graph[x].append(y)
            graph[y].append(x)
        visited = set()
        connected = 0
        def bfs(i):
            queue = deque([i])
            while queue:
                i = queue.pop()
                for j in graph[i]:
                    if j not in visited:
                        visited.add(j)
                        queue.append(j)
        for i in range(n):
            if i not in visited:
                connected+=1
                visited.add(i)
                bfs(i)
        return connected
"""DFS"""
class Solution:
    def countComponents(self, n: int, edges: List[List[int]]) -> int:
        from collections import defaultdict, deque
        graph = defaultdict(list)
        for x, y in edges:
            graph[x].append(y)
            graph[y].append(x)
        visited = set()
        connected = 0
        def dfs(i):
            visited.add(i)
            for j in graph[i]:
                if j not in visited:
                    dfs(j)
        for i in range(n):
            if i not in visited:
                connected+=1
                dfs(i)
        return connected
"""
2, 二维平面上的搜索问题
"""
#695，岛屿的最大面积
#给定一个包含了一些0和1的费控二维数组grid,一个岛屿是由一些相邻的1（代表土地）
#构成的组合，这里的相邻要求两个1必须在水平或者竖直方向上相邻，可以假设grid的
#四个边缘都被0（代表水）包围着，找到给定的二维数组中最大的岛屿面积（如果没有返回0）
"""BFS"""
class Solution:
    def maxAreaOfIsland(self, grid: List[List[int]]) -> int:
        from collections import deque
        ans = 0
        for i, l in enumerate(grid):
            for j,n in enumerate(l):
                cur = 0
                q = deque([(i,j)])
                while q:
                    cur_i, cur_j = q.popleft()
                    # put all lands into queue and pass water
                    if cur_i<0 or cur_j<0 or cur_i == len(grid) or cur_j ==len(grid[0]) or grid[cur_i][cur_j] !=1:
                        continue
                    cur +=1
                    grid[cur_i][cur_j] = 0
                    for di, dj in [[0,1],[0,-1],[1,0],[-1,0]]:
                        next_i, next_j = cur_i+di, cur_j+dj
                        q.append((next_i,next_j))
                ans = max(ans,cur)
        return ans
"""DFS"""
# Time complex: O(R*C),R是给定的网格中的行数, C是列数，我们访问每个网格最
# Space complex: O(R*C),
class Solution:
    def maxAreaOfIsland(self, grid: List[List[int]]) -> int:
        from collections import deque
        ans = 0
        def dfs(self,grid,cur_i,cur_j):
            if cur_i < 0 or cur_j < 0 or cur_i == len(grid) or cur_j == len(grid[0]) or grid[cur_i][cur_j] != 1:
                return 0
            grid[cur_i][cur_j] = 0
            ans =1
            for di, dj in [[0,1],[0,-1],[1,0],[-1,0]]:
                next_i, next_j = cur_i + di, cur_j + dj
                ans += self.dfs(grid,next_i,next_j)
            return ans
        for i, l in enumerate(grid):
            for j,n in enumerate(l):
                ans = max(self.dfs(grid,i,j),ans)
        return ans
""""
三，排序
1, 时间复杂度 O(n^2) 级排序算法
1）冒泡排序
是入门级的算法，但也有一些有趣的玩法。通常来说，冒泡排序有三种写法：
一边比较一边向后两两交换，将最大值 / 最小值冒泡到最后一位；
经过优化的写法：使用一个变量记录当前轮次的比较是否发生过交换，如果没有发生交换表示已经有序，不再继续排序；
进一步优化的写法：除了使用变量记录当前轮次是否发生交换外，再使用一个变量记录上次发生交换的位置，下一轮排序时到达上次交换的位置就停止比较。
2, 时间复杂度 O(nlogn) 级排序算法
1）希尔排序
虽然原始的希尔排序最坏时间复杂度仍然是 O(n^2)，但经过优化的希尔排序可以达到 O(n^{1.3}) 甚至 O(n^{7/6})
希尔排序本质上是对插入排序的一种优化，它利用了插入排序的简单，又克服了插入排序每次只交换相邻两个元素的缺点。它的基本思想是：
将待排序数组按照一定的间隔分为多个子数组，每组分别进行插入排序。这里按照间隔分组指的不是取连续的一段数组，而是每跳跃一定间隔取一个值组成一组
逐渐缩小间隔进行下一轮排序
最后一轮时，取间隔为 11，也就相当于直接使用插入排序。但这时经过前面的「宏观调控」，数组已经基本有序了，所以此时的插入排序只需进行少量交换便可完成
2）堆排序
堆：符合以下两个条件之一的完全二叉树：
根节点的值 ≥ 子节点的值，这样的堆被称之为最大堆，或大顶堆；
根节点的值 ≤ 子节点的值，这样的堆被称之为最小堆，或小顶堆。
堆排序过程如下：
用数列构建出一个大顶堆，取出堆顶的数字；
调整剩余的数字，构建出新的大顶堆，再次取出堆顶的数字；
循环往复，完成整个排序。
整体的思路就是这么简单，我们需要解决的问题有两个：
如何用数列构建出一个大顶堆；
取出堆顶的数字后，如何将剩余的数字调整成新的大顶堆。
构建大顶堆 & 调整堆
构建大顶堆有两种方式：
方案一：从 0 开始，将每个数字依次插入堆中，一边插入，一边调整堆的结构，使其满足大顶堆的要求；
方案二：将整个数列的初始状态视作一棵完全二叉树，自底向上调整树的结构，使其满足大顶堆的要求。
第二种方案更加常用。
在介绍堆排序具体实现之前，我们先要了解完全二叉树的几个性质。将根节点的下标视为 0，则完全二叉树有如下性质：
假设一个二叉树的深度为h，除第 h 层外，其它各层 (1～h-1) 的结点数都达到最大个数，第 h 层所有的结点都连续集中在最左边，这就是完全二叉树
对于完全二叉树中的第 i 个数，它的左子节点下标：left = 2i + 1
对于完全二叉树中的第 i 个数，它的右子节点下标：right = left + 1
对于有 n 个元素的完全二叉树(n≥2)(n≥2)，它的最后一个非叶子结点的下标：n/2 - 1
"""
def heapSort(arr):
    # 构建初始大顶堆
    build_max_heap(arr)
    lenth = len(arr)
    for i in range(lenth-1, 0, -1):
        # 将最大值交换到数组最后
        swap(arr,0,i)
        # 调整剩余数组，使其满足大顶堆
        max_heapify(arr,0,i)
# 桂建初始大顶堆
def build_max_heap(arr):
    n = len(arr)
    for i in range(n/2-1,0,-1):
        max_heapify(arr,i,n)
# 调整大顶堆，第三个参数表示剩余未排序的数量，也是剩余堆的大小
def max_heapify(arr, i, heap_size):
    # 左子节点下标
    l = 2*i+1
    # 右子节点下标
    r = l+1
    largest = i
    # 与左子树节点比较
    if heap_size >1 and arr[l]>arr[largest]:
        largest = l
    # 与右子树节点比较
    if heap_size >1 and arr[r]>arr[largest]:
        largest = r
    if largest != i:
        # 将最大值交换为根节点
        swap(arr, i, largest)
        # 再次调整交换后的最大顶堆
        max_heapify(arr,largest,heap_size)
def swap(arr, i, j):
    temp = arr[i]
    arr[i] = arr[j]
    arr[j] = arr[i]
"""2, 时间复杂度 O(nlogn) 级排序算法
3）快排
时间复杂度O(nlogn)，时间复杂度也是O(nlogn)
基本思想：
step1: 从数组中取出一个数，称之为基数(pivot)
step2: 遍历数组，将比基数大的数字放在其右边，比基数小的放在左边，遍历完成，数组被分成左右两个区域
step3: 将两个区域视为两个数组，重复前面两个步骤，直到完成为止
"""
class Solution:
    def sortArray(self, nums: List[int]) -> List[int]:
        if not nums or len(nums) == 0:
            return
        self.quickSort(nums, 0, len(nums) - 1)
        return nums
    def quickSort(self, nums, start, end):
        if start >= end:
            return
        left, right = start, end
        pivot = (nums[left] + nums[right]) / 2
        while left <= right:
            while left <= right and nums[left] < pivot:
                left += 1
            while left <= right and nums[right] > pivot:
                right -= 1
            if left <= right:
                nums[right], nums[left] = nums[left], nums[right]
                left += 1
                right -= 1
        self.quickSort(nums, start, right)
        self.quickSort(nums, left, end)
#347, Top K Frequent Elements
class Solution:
    def findKthLargest(self, nums: List[int], k: int) -> int:
        if not nums or len(nums) == 0:
            return
        self.qucikSelect(nums, 0, len(nums) - 1, k)
        return nums[k - 1]
    def qucikSelect(self, nums, start, end, k):
        if start >= end:
            return
        left, right = start, end
        pivot = (nums[left] + nums[right]) / 2
        while left <= right:
            while left <= right and nums[left] > pivot:
                left += 1
            while left <= right and nums[right] < pivot:
                right -= 1
            if left <= right:
                nums[right], nums[left] = nums[left], nums[right]
                left += 1
                right -= 1
        if start + k - 1 <= right:
            self.qucikSelect(nums, start, right, k)
        if start + k - 1 >= left:
            self.qucikSelect(nums, left, end, k - (left - start))
""""
四，二叉树
二叉树的遍历问题或者其他的任何问题，都考虑把它落实在每个子树上，然后在每颗字数上考虑，推广到全局
1，二叉树的前/中/后序遍历问题
"""
# 144. Binary Tree Preorder Traversal。 Given a binary tree, return the preorder traversal of its nodes' values
class TreeNode:
    def __init__(self,x):
        self.val = x
        self.left = None
        self.right = None
"""非递归的方式--使用stack的方法：因为栈先进后出，可以pop右边的能力"""
class Solution:
    def preorderTraversal(self,root:TreeNode): # root,left,right
        if not root:
            return []
        stack = [root]
        res = []
        while stack:
            node = stack.pop()
            res.append(node.val)
            if node.right:
                stack.append(node.right)
            if node.left:
                stack.append(node.left)
        return res
"""递归的方式--使用递归方法"""
class Solution:
    def preorderTraversal(self,root:TreeNode): # root,left,right
        if not root:
            return []
        res = []
        self.traverse(root,res)
        return res
    def traverse(self,root,res):
        if not root:
            return
        res.append(root.val)
        self.traverse(root.left,res)
        self.traverse(root.right,res)
# 94. Binary Tree Inorder Traversal.  Given a binary tree, return the inorder traversal of its nodes' values
"""递归的方式--使用递归方法"""
class Solution:
    def inorderTraversal(self,root:TreeNode): # left,root,right
        if not root:
            return []
        res = []
        self.traverse(root,res)
        return res
    def traverse(self,root,res):
        if not root:
            return
        self.traverse(root.left,res)
        res.append(root.val)
        self.traverse(root.right,res)
"""非递归的方式--使用stack的方式"""
class Solution:
    def inorderTraversal(self,root:TreeNode): # left,root,right
        if not root:
            return []
        stack = []
        res = []
        while root:
            res.append(root)
            root=root.left
        while stack:
            node =stack.pop()
            res.append(node.val)
            if node.right:
                node1 = node.right
                while node1:
                    stack.append(node1)
                    node1 = node1.left
        return res
#144. Binary Tree Preorder Traversal.Given a binary tree, return the preorder traversal of its nodes' values
"""递归的方式--使用递归方法"""
class Solution:
    def postorderTraversal(self, root: TreeNode):# left,right,root
        if not root:
            return []
        res = []
        self.traverse(root, res)
        return res
    def traverse(self, root, res):
        if not root:
            return
        self.traverse(root.left, res)
        self.traverse(root.right, res)
        res.append(root.val)
#105. Construct Binary Tree from Preorder and Inorder Traversal
class Solution:
    def reConstructBinaryTree(self, pre, tin):
        if not pre or not tin:
            return None
        rootVal = pre[0]
        id = tin.index(pre[0])
        root = TreeNode(rootVal)
        root.left = self.reConstructBinaryTree(pre[1:id+1],tin[:id])
        root.right = self.reConstructBinaryTree(pre[id+1:],tin[id+1:])
        return root
""""
四，二叉树的基本变化序
1, DFS遍历，一般不需要返回值。但是可以在遍历的过程中进行各种操作，以达到目的
"""
# 226. Invert Binary Tree
class TreeNode:
    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None
class Solution:
    def invertTree(self,root):
        if not root:
            return None
        self.dfs(root)
        return root
    def dfs(self,root):
        if not root:
            return
        left =root.left
        right = root.right
        root.left = right
        root.right = left
        if root.right:
            self.dfs(root.right)
        if root.left:
            self.dfs(root.left)
""""
2，分治法：分治法一般要返回值，为了求最大值，需要问题转化成解决左子树上深度，
和右子树深度。然后合并中最大的就是全局深度。因为每次遍历深度加一。
"""
#101. Symmetric Tree。Given a binary tree, check whether it is
# a mirror of itself (ie, symmetric around its center。
class TreeNode:
    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None
class Solution:
    def isSymmetric(self, root: TreeNode):
        if not root:
            return True
        return self.checkSymmetric(root.left,root.right)
    def checkSymmetric(self,nodeA,nodeB):
        if nodeA is None or nodeB is None:
            return False
        if nodeA is None and nodeB is None:
            return True
        if nodeA.val != nodeB.val:
            return False
        inner_res = self.checkSymmetric(nodeA.left,nodeB.right)
        outer_res = self.checkSymmetric(nodeA.right, nodeB.left)
        return inner_res and outer_res
""""
3，Path Sum类型问题一般分为三种方式:
1) 判断是否存在从root- leaf 和为指定的数字的路径
（1）构造一个递归出口，一般就是在root 为空时候
（2）构造一个到leaf节点时候 找到指定路径的情况，
    一般就是，当root.left和root.right为空，root.val == 此时target数字
（3）然后进行左右递归，注意，参数中的，root和target数字都是要需要更新的
"""
#112. Path Sum. Given a binary tree and a sum, determine if the tree has a root-to-leaf path such that
# adding up all the values along the path equals the given sum.
class TreeNode:
    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None
class Solution:
    def hasPathSum(self,root,sum):
        if root is None:
            return False
        if root.left is None and root.left is None:
            return sum == root.val
        return self.hasPathSum(root.left, sum-root.val) or self.hasPathSum(root.right, sum-root.val)
#437. Path Sum III. You are given a binary tree in which each node contains an integer value
#Find the number of paths that sum to a given value.The path does not need to start or end at
#the root or a leaf, but it must go downwards (traveling only from parent nodes to child nodes)
#The tree has no more than 1,000 nodes and the values are in the range -1,000,000 to 1,000,000
class TreeNode:
    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None
class Solution:
    def pathSum(self, root: TreeNode, sum: int):
 # 思路分析： 因为题目中要求，只要path之和等于所给的sum，可以不必须是从root开始，而是可以从任何地方开始。
 # 那么就需要考虑，root, root.left, root.right 三种情况下的。总的path的数量就是他们三个所包含的之和
        if not root:
            return 0
        res = []
        self.helper(root, [], sum, res)
        return res
    def helper(self, root, path, target, res):
        if not root:
            return
        path += [root.val]
        flag = (root.left == None and root.right === None)
        if root.val ==target and flag:
            res.append(path[:])
            return
        if root.left:
            self.helper(root.left, path[:], target-root.val,res)
        if root.right:
            self.helper(root.right, path[:], target-root.val,res)
        path.pop()
"""
4，关于最大最小长度问题：一般采用的是分治法
(1) 一般需要返回三类值:1)curLen/curSum, 返回当前所求的长度和值；2）maxLen/maxSum，返回最值; 3) node或者nodeVal,
返回节点或者节点数值，具体选择哪一个，看题中要求；
(2) 对于空集的判断if not root,需要注意的是，最值在初始化的时候，最大值一般是用-sys.maxsizem,最小值是用sys.maxsize
(3) divide，分别对左右进行循环
(4) conque, 在conque的时候，需要分两步:1)先根据调解，求出curLen/curSum；2）然后在判断出最值
"""
# Minimum Subtree.Given a binary tree, find the subtree with minimum sum. Return the root of the subtree
class TreeNode:
    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None
class Solution:
    def findSubtree(self, root: TreeNode):
        if not root:
            return None
        node, minSubtree,Sum = self.helper(root)
        return node
    def helper(self,root):
        if not root:
            return None, 0, sys.maxsize
        left_node, left_min_sum, left_sum = self.helper(root.left)
        right_node, right_min_sum, right_sum = self.helper(root.right)
        total_sum = left_sum + root.val+ right_sum
        if left_min_sum == min(left_min_sum, right_min_sum, total_sum):
            return left_node, left_min_sum, total_sum
        if right_min_sum == min(left_min_sum, right_min_sum, total_sum):
            return right_node , right_min_sum, total_sum
        if total_sum == min(left_min_sum, right_min_sum, total_sum):
            return root, total_sum, total_sum
# 298. Binary Tree Longest Consecutive Sequence.Given a binary tree, find the length of the longest consecutive sequence path
# The path refers to any sequence of nodes from some starting node to any node in the tree along the parent-child connections.
# The longest consecutive path need to be from parent to child (cannot be the reverse)
class TreeNode:
    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None
class Solution:
    def longestConsecutive(self, root: TreeNode):
        if not root:
            return 0
        maxLen, curLen, curVal = self.helper(root)
        return maxLen
    def helper(self, root):
        if not root:
            return -sys.maxsize, 0, 0
        left_maxLen, left_curLen, left_curVal = self.helper(root.left)
        right_maxLen, right_curLen, right_curVal = self.helper(root.right)
        curLen =1
        if root.val == left_curVal -1:
            curLen = max(curLen, left_curLen+1)
        if root.val == right_curVal -1:
            curLen = max(curLen, right_curLen+1)
        if left_maxLen == max(curLen, left_maxLen, right_maxLen):
            return left_maxLen, curLen, root.val
        if right_maxLen == max(curLen, left_maxLen, right_maxLen):
            return right_maxLen, curLen, root.val
        return curLen,curLen,root.val
"""
5, 二叉树宽度有限搜索BFS：主要是使用deque这类函数先进先出的优点，配合deque.popleft(), 实现层序遍历
"""
# 102. Binary Tree Level Order Traversal。Given a binary tree, return the level order traversal
# of its nodes' values. (ie, from left to right, level by level)
class TreeNode:
    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None
class Solution:
    def levelOrder(self, root: TreeNode):
        if not root:
            return []
        queue = deque([root])
        res = []
        while queue:
            level = []
            for _ in range(len(queue)):
                node = queue.popleft()
                level.append(node.val)
                if node.left:
                    queue.append(node.left)
                if node.right:
                    queue.append(node.right)
            res.append(level)
        return res
"""一,双指针
时间复杂度一般是O(n)
Slicing Windows and two points
什么是滑动窗口？
其实就是一个队列,比如例题中的 abcabcbb，进入这个队列（窗口）为 abc 满足题目要求，当再进入 a，队列变成了 abca，这时候不满足要求。
所以，我们要移动这个队列！
如何移动？
我们只要把队列的左边的元素移出就行了，直到满足题目要求！
一直维持这样的队列，找出队列出现最长的长度时候，求出解！
时间复杂度：O(n)
1) 经典双指针使用统计非重复数量
"""
#3. Longest Substring Without Repeating Characters，Given a string, find the length of the
#longest substring without repeating characters
class Solution:
    def lengthOfLongestSubstring(self, s):
        if not s:
            return 0
        left = 0
        # the left boundary of window
        curLen, maxLen = 0, 0
        visited = set()
        n = len(s)
        for i in range(n):
            curLen += 1
            while s[i] in visited:
                visited.remove(s[left])
                curLen -= 1
                left += 1
            if maxLen < curLen:
                maxLen = curLen
            visited.add(s[i])
        return maxLen
"""
2) 同一个数组，要in-place的去掉里面的特殊数，类似0，或者重复
Step1: 设置双指针，同一起点，注意如果有特殊需求，类似不可出现两次这样的，起点设置不一定是从0开始
Step2: 让右指针作为大循环，然后在里面先判断不满足条件的情况，让左右指针的值互换，然后左指针前行，为小循环
Step3: 返回值，注意返回值是否加上1
"""
#283. Move Zeroes，Given an array nums, write a function to move all 0's to the end of it
# while maintaining the relative order of the non-zero elements.
class Solution:
    def moveZeros(self, nums):
        left, right = 0, 0
        n = len(nums)
        while right < n:
            if nums[right] !=0:
                if left!=right:
                    nums[left] = nums[right]
                left +=1
            right +=1
        while left < n:
            if nums[left] !=0:
               nums[left] = 0
            left +=1
        return nums
# 604, Window Sum. Given an array of n integers, and a moving window(size k), move the window at each",
# iteration from the start of the array, find the sum of the element inside the window at",
# each moving
class Solution:
    def winSum(self,nums,k):
        if not nums or k <0:
            return []
        left, right = 0, k-1
        res = []
        while right < 0:
            if left == 0:
                sum_each = sum(nums[0:right+1])
                res.append(sum_each)
            else:
                sum_each = res[-1] - nums[left-1]+nums[right]
                res.append(sum_each)
            left+=1
            right+=1
        return res
#287. Find the Duplicate Number。 Given an array nums containing n + 1 integers where each integer is
# between 1 and n (inclusive), prove that at least one duplicate number must exist. Assume that there
# is only one duplicate number, find the duplicate one
"""使用类似环形链表142题的方式，
关键是理解如何把输入的数组看作为链表。首先要明确前提，整个数组中的nums是在[1,n]之间，考虑两种情况：
1) 如果数组中没有重复的数字，以数组[1,3.4.2]为例，我们将数组下标n和数nums[n]建立映射关系f(n), 其隐射关系n->f(n)为：
0->1 
1->3
2->4
3->2
我们以下标0出发，根据f(n)计算出一个值，以这个值为新的下标，再用这个函数计算，以此类推，直到下标超界。这样就可以产生一个类似链表的
序列0->1->3->2->4->null
2) 如果数组中有重复的数字，以数组[1,3.4.2，2]为例，我们将数组下标n和数nums[n]建立映射关系f(n), 其隐射关系n->f(n)为：
0->1 
1->3
2->4
3->2
4->2
同样可以构造链表：0->1->3->2->4->3->2->4->2，从理论上讲，数组中如果有重复的数，那么就会产生多对一的映射，这样，形成的链表就一定会有环路了
综述，如果数组中有重复数字，那么链表中就存在环，找到环的入口就是找到这个数字
142中慢指针走一步 slow=slow.next 等于本题中slow=nums[slow]
142中快指针走一步 fast=fast.next 等于本题中fast=nums[nums[fast]]
"""
class Solutions:
    def findDulicate(self,nums):
        if len(nums)<=1:
            return -1
        slow = nums[0]
        fast = nums[nums[0]]
        while slow!=fast:
            slow = nums[slow]
            fast= nums[nums[fast]]
        fast = 0
        while fast!=slow:
            fast = nums[fast]
            slow = nums[slow]
        return slow
# 142. Linked List Cycle II.Given a linked list, return the node where the cycle begins.
# If there is no cycle, return null.To represent a cycle in the given linked list, we use
# an integer pos which represents the position (0-indexed) in the linked list where tail
# connects to. If pos is -1, then there is no cycle in the linked list. Note: Do not modify the linked list
class Solutions:
    def detectCycle(self, head):
        if not head or head.next:
            return -1
        slow, fast = head, head
        while fast and fast.next:
            slow = slow.next
            fast = fast.next.next
            if slow is fast:
                break
        if fast is slow:
            slow = head
            while fast is not slow:
                fast = fast.next
                slow = slow.next
            return slow
        return None
# 2. Add Two Numbers. You are given two non-empty linked lists representing two non-negative integers.
# The digits are stored in reverse order and each of their nodes contain a single digit. Add the two numbers and return it as a linked list.",
class Solutions:
    def addTwoNumbers(self,l1:ListNode,l2:ListNode):
        #首先创建一个虚拟节点，并创建一个current指针，指向这个节点
        current = dummy = ListNode()
        #初始化carry和两个链表对应节点相加的值
        carry, value = 0, 0
        #下面的while循环中之所以有carry，是为了处理两个链表最后节点相加出现进位的情况
        #当两个节点都走完而且最后的运算并没有进位时，就不会进入这个循环
        while carry or l1 or l2:
            #让value先等于carry既有利于下面两个if语句中两个对应节点值相加，
            # 也是为了要处理两个链表最后节点相加出现进位的情况
            value = carry
            #只要其中一个链表没走完，就需要计算value的值
            #如果其中一个链表走完，那么下面的计算就是加总carry和其中一个节点的值
            #如果两个链表都没走完，那么下面的计算就是carry+对应的两个节点的值
            if l1: l1, value = l1.next, l1.val + value
            if l2: l2, value = l2.next, l2.val + value
            #为了防止value值大于十，出现进位，需要特殊处理
            #如果value小于十，下面这行的操作对于carry和value的值都没有影响
            carry, value = divmod(value, 10)
            #利用value的值创建一个链表节点，并让current.next指向它
            current.next = ListNode(value)
            #移动current指针到下一个节点
            current = current.next
        #最后只要返回dummy的下一个节点就是我们想要的答案。
        return dummy.next
""""
二，BFS
1, 齐头并进的广度优先遍历问题：
(1) 树的宽度优先遍历
"""
# 102. Binary Tree Level Order Traversal
# Given a binary tree, return the level order traversal of its nodes' values. (ie, from left to right, level by level
class TreeNode:
    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None
from collections import deque
class Solution:
    def levelOrder(self,root:TreeNode) -> List[List[int]]:
        if not root:
            return []
        queue = deque([root])
        res =[]
        while queue:
            level =[]
            for _ in range(len(queue)):
                node = queue.popleft()
                level.append(node.val)
                if node.left:
                    queue.append(node.left)
                if node.right:
                    queue.append(node.right)
            res.append(level)
        return res
# 107. Binary Tree Level Order Traversal II
# Given a binary tree, return the bottom-up level order traversal of its nodes' values. (ie, from left to right, level by level from leaf to right
class TreeNode:
    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None
from collections import deque
class Solution:
    def levelOrderBottom(self, root:TreeNode) -> List[List[int]]:
        if not root:
            return []
        queue = deque([root])
        res = []
        while queue:
            level = []
            for _ in range(len(queue)):
                node = queue.popleft()
                level.append(node.val)
                if node.left:
                    queue.append(node.left)
                if node.right:
                    queue.append(node.right)
            res.insert(0,level)
        return res
# 103. Binary Tree Zigzag Level Order Traversal
# Given a binary tree, return the zigzag level order traversal of its nodes' values. (ie, from left to right, then right to left for the next level and alternate between)
class TreeNode:
    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None
from collections import deque
class Solution:
    def zigzagLevelOrder(self, root: TreeNode) -> List[List[int]]:
        if not root:
            return []
        queue = deque([root])
        count =0
        res = []
        while queue:
            count +=1
            level =[]
            for _ in range(len(queue)):
                node = queue.popleft()
                if count%2 ==1:
                    level.append(node.val)
                else:
                    level.insert(0,node.val)
                if node.left:
                    queue.append(node.left)
                if node.right:
                    queue.append(node.right)
            res.append(level)
        return res
"""
（2）无权图的最短路径
--依赖的数据结构：队列
--应用：无权图中找到最短路径
--注意事项：无权图中遍历，加入队列之后，必须马上标记【已经访问】
在图中，由于 图中存在环，和深度优先遍历一样，广度优先遍历也需要在遍历的时候记录已经遍历过的结点。
特别注意：将结点添加到队列以后，一定要马上标记为「已经访问」，否则相同结点会重复入队，
这一点在初学的时候很容易忽略。如果很难理解这样做的必要性，建议大家在代码中打印出队列中的元素进行调试：
在图中，如果入队的时候不马上标记为「已访问」，相同的结点会重复入队，这是不对的。另外一点还需要强调，
广度优先遍历用于求解「无权图」的最短路径，因此一定要认清「无权图」这个前提条件。
如果是带权图，就需要使用相应的专门的算法去解决它们
"""
# 323. 无向图中连通分量的数目
# 给定编号从 0 到 n-1 的 n 个节点和一个无向边列表（每条边都是一对节点），请编写一个函数来计算无向图中连通分量的数目。
"""BFS"""
class Solution:
    def countComponents(self, n: int, edges: List[List[int]]) -> int:
        from collections import defaultdict, deque
        graph = defaultdict(list)
        #只有当list里面是两个元素才可以使用这样的方式
        for x, y in edges:
            graph[x].append(y)
            graph[y].append(x)
        visited = set()
        connected = 0
        def bfs(i):
            queue = deque([i])
            while queue:
                i = queue.pop()
                for j in graph[i]:
                    if j not in visited:
                        visited.add(j)
                        queue.append(j)
        for i in range(n):
            if i not in visited:
                connected+=1
                visited.add(i)
                bfs(i)
        return connected
"""DFS"""
class Solution:
    def countComponents(self, n: int, edges: List[List[int]]) -> int:
        from collections import defaultdict, deque
        graph = defaultdict(list)
        for x, y in edges:
            graph[x].append(y)
            graph[y].append(x)
        visited = set()
        connected = 0
        def dfs(i):
            visited.add(i)
            for j in graph[i]:
                if j not in visited:
                    dfs(j)
        for i in range(n):
            if i not in visited:
                connected+=1
                dfs(i)
        return connected
"""
2, 二维平面上的搜索问题
"""
#695，岛屿的最大面积
#给定一个包含了一些0和1的费控二维数组grid,一个岛屿是由一些相邻的1（代表土地）
#构成的组合，这里的相邻要求两个1必须在水平或者竖直方向上相邻，可以假设grid的
#四个边缘都被0（代表水）包围着，找到给定的二维数组中最大的岛屿面积（如果没有返回0）
"""BFS"""
class Solution:
    def maxAreaOfIsland(self, grid: List[List[int]]) -> int:
        from collections import deque
        ans = 0
        for i, l in enumerate(grid):
            for j,n in enumerate(l):
                cur = 0
                q = deque([(i,j)])
                while q:
                    cur_i, cur_j = q.popleft()
                    # put all lands into queue and pass water
                    if cur_i<0 or cur_j<0 or cur_i == len(grid) or cur_j ==len(grid[0]) or grid[cur_i][cur_j] !=1:
                        continue
                    cur +=1
                    grid[cur_i][cur_j] = 0
                    for di, dj in [[0,1],[0,-1],[1,0],[-1,0]]:
                        next_i, next_j = cur_i+di, cur_j+dj
                        q.append((next_i,next_j))
                ans = max(ans,cur)
        return ans
"""DFS"""
# Time complex: O(R*C),R是给定的网格中的行数, C是列数，我们访问每个网格最
# Space complex: O(R*C),
class Solution:
    def maxAreaOfIsland(self, grid: List[List[int]]) -> int:
        from collections import deque
        ans = 0
        def dfs(self,grid,cur_i,cur_j):
            if cur_i < 0 or cur_j < 0 or cur_i == len(grid) or cur_j == len(grid[0]) or grid[cur_i][cur_j] != 1:
                return 0
            grid[cur_i][cur_j] = 0
            ans =1
            for di, dj in [[0,1],[0,-1],[1,0],[-1,0]]:
                next_i, next_j = cur_i + di, cur_j + dj
                ans += self.dfs(grid,next_i,next_j)
            return ans
        for i, l in enumerate(grid):
            for j,n in enumerate(l):
                ans = max(self.dfs(grid,i,j),ans)
        return ans
""""
三，排序
1, 时间复杂度 O(n^2) 级排序算法
1）冒泡排序
是入门级的算法，但也有一些有趣的玩法。通常来说，冒泡排序有三种写法：
一边比较一边向后两两交换，将最大值 / 最小值冒泡到最后一位；
经过优化的写法：使用一个变量记录当前轮次的比较是否发生过交换，如果没有发生交换表示已经有序，不再继续排序；
进一步优化的写法：除了使用变量记录当前轮次是否发生交换外，再使用一个变量记录上次发生交换的位置，下一轮排序时到达上次交换的位置就停止比较。
2, 时间复杂度 O(nlogn) 级排序算法
1）希尔排序
虽然原始的希尔排序最坏时间复杂度仍然是 O(n^2)，但经过优化的希尔排序可以达到 O(n^{1.3}) 甚至 O(n^{7/6})
希尔排序本质上是对插入排序的一种优化，它利用了插入排序的简单，又克服了插入排序每次只交换相邻两个元素的缺点。它的基本思想是：
将待排序数组按照一定的间隔分为多个子数组，每组分别进行插入排序。这里按照间隔分组指的不是取连续的一段数组，而是每跳跃一定间隔取一个值组成一组
逐渐缩小间隔进行下一轮排序
最后一轮时，取间隔为 11，也就相当于直接使用插入排序。但这时经过前面的「宏观调控」，数组已经基本有序了，所以此时的插入排序只需进行少量交换便可完成
2）堆排序
堆：符合以下两个条件之一的完全二叉树：
根节点的值 ≥ 子节点的值，这样的堆被称之为最大堆，或大顶堆；
根节点的值 ≤ 子节点的值，这样的堆被称之为最小堆，或小顶堆。
堆排序过程如下：
用数列构建出一个大顶堆，取出堆顶的数字；
调整剩余的数字，构建出新的大顶堆，再次取出堆顶的数字；
循环往复，完成整个排序。
整体的思路就是这么简单，我们需要解决的问题有两个：
如何用数列构建出一个大顶堆；
取出堆顶的数字后，如何将剩余的数字调整成新的大顶堆。
构建大顶堆 & 调整堆
构建大顶堆有两种方式：
方案一：从 0 开始，将每个数字依次插入堆中，一边插入，一边调整堆的结构，使其满足大顶堆的要求；
方案二：将整个数列的初始状态视作一棵完全二叉树，自底向上调整树的结构，使其满足大顶堆的要求。
第二种方案更加常用。
在介绍堆排序具体实现之前，我们先要了解完全二叉树的几个性质。将根节点的下标视为 0，则完全二叉树有如下性质：
对于完全二叉树中的第 i 个数，它的左子节点下标：left = 2i + 1
对于完全二叉树中的第 i 个数，它的右子节点下标：right = left + 1
对于有 n 个元素的完全二叉树(n≥2)(n≥2)，它的最后一个非叶子结点的下标：n/2 - 1
"""
def heapSort(arr):
    # 构建初始大顶堆
    build_max_heap(arr)
    lenth = len(arr)
    for i in range(lenth-1, 0, -1):
        # 将最大值交换到数组最后
        swap(arr,0,i)
        # 调整剩余数组，使其满足大顶堆
        max_heapify(arr,0,i)
# 桂建初始大顶堆
def build_max_heap(arr):
    n = len(arr)
    for i in range(n/2-1,0,-1):
        max_heapify(arr,i,n)
# 调整大顶堆，第三个参数表示剩余未排序的数量，也是剩余堆的大小
def max_heapify(arr, i, heap_size):
    # 左子节点下标
    l = 2*i+1
    # 右子节点下标
    r = l+1
    largest = i
    # 与左子树节点比较
    if heap_size >1 and arr[l]>arr[largest]:
        largest = l
    # 与右子树节点比较
    if heap_size >1 and arr[r]>arr[largest]:
        largest = r
    if largest != i:
        # 将最大值交换为根节点
        swap(arr, i, largest)
        # 再次调整交换后的最大顶堆
        max_heapify(arr,largest,heap_size)
def swap(arr, i, j):
    temp = arr[i]
    arr[i] = arr[j]
    arr[j] = arr[i]
"""2, 时间复杂度 O(nlogn) 级排序算法
3）快排
时间复杂度O(nlogn)，时间复杂度也是O(nlogn)
基本思想：
step1: 从数组中取出一个数，称之为基数(pivot)
step2: 遍历数组，将比基数大的数字放在其右边，比基数小的放在左边，遍历完成，数组被分成左右两个区域
step3: 将两个区域视为两个数组，重复前面两个步骤，直到完成为止
"""
class Solution:
    def sortArray(self, nums: List[int]) -> List[int]:
        if not nums or len(nums) == 0:
            return
        self.quickSort(nums, 0, len(nums) - 1)
        return nums
    def quickSort(self, nums, start, end):
        if start >= end:
            return
        left, right = start, end
        pivot = (nums[left] + nums[right]) / 2
        while left <= right:
            while left <= right and nums[left] < pivot:
                left += 1
            while left <= right and nums[right] > pivot:
                right -= 1
            if left <= right:
                nums[right], nums[left] = nums[left], nums[right]
                left += 1
                right -= 1
        self.quickSort(nums, start, right)
        self.quickSort(nums, left, end)
#347, Top K Frequent Elements
class Solution:
    def findKthLargest(self, nums: List[int], k: int) -> int:
        if not nums or len(nums) == 0:
            return
        self.qucikSelect(nums, 0, len(nums) - 1, k)
        return nums[k - 1]
    def qucikSelect(self, nums, start, end, k):
        if start >= end:
            return
        left, right = start, end
        pivot = (nums[left] + nums[right]) / 2
        while left <= right:
            while left <= right and nums[left] > pivot:
                left += 1
            while left <= right and nums[right] < pivot:
                right -= 1
            if left <= right:
                nums[right], nums[left] = nums[left], nums[right]
                left += 1
                right -= 1
        if start + k - 1 <= right:
            self.qucikSelect(nums, start, right, k)
        if start + k - 1 >= left:
            self.qucikSelect(nums, left, end, k - (left - start))
""""
四，二叉树
二叉树的遍历问题或者其他的任何问题，都考虑把它落实在每个子树上，然后在每颗字数上考虑，推广到全局
1，二叉树的前/中/后序遍历问题
"""
# 144. Binary Tree Preorder Traversal。 Given a binary tree, return the preorder traversal of its nodes' values
class TreeNode:
    def __init__(self,x):
        self.val = x
        self.left = None
        self.right = None
"""非递归的方式--使用stack的方法：因为栈先进后出，可以pop右边的能力"""
class Solution:
    def preorderTraversal(self,root:TreeNode): # root,left,right
        if not root:
            return []
        stack = [root]
        res = []
        while stack:
            node = stack.pop()
            res.append(node.val)
            if node.right:
                stack.append(node.right)
            if node.left:
                stack.append(node.left)
        return res
"""递归的方式--使用递归方法"""
class Solution:
    def preorderTraversal(self,root:TreeNode): # root,left,right
        if not root:
            return []
        res = []
        self.traverse(root,res)
        return res
    def traverse(self,root,res):
        if not root:
            return
        res.append(root.val)
        self.traverse(root.left,res)
        self.traverse(root.right,res)
# 94. Binary Tree Inorder Traversal.  Given a binary tree, return the inorder traversal of its nodes' values
"""递归的方式--使用递归方法"""
class Solution:
    def inorderTraversal(self,root:TreeNode): # left,root,right
        if not root:
            return []
        res = []
        self.traverse(root,res)
        return res
    def traverse(self,root,res):
        if not root:
            return
        self.traverse(root.left,res)
        res.append(root.val)
        self.traverse(root.right,res)
"""非递归的方式--使用stack的方式"""
class Solution:
    def inorderTraversal(self,root:TreeNode): # left,root,right
        if not root:
            return []
        stack = []
        res = []
        while root:
            res.append(root)
            root=root.left
        while stack:
            node =stack.pop()
            res.append(node.val)
            if node.right:
                node1 = node.right
                while node1:
                    stack.append(node1)
                    node1 = node1.left
        return res
#144. Binary Tree Preorder Traversal.Given a binary tree, return the preorder traversal of its nodes' values
"""递归的方式--使用递归方法"""
class Solution:
    def postorderTraversal(self, root: TreeNode):# left,right,root
        if not root:
            return []
        res = []
        self.traverse(root, res)
        return res
    def traverse(self, root, res):
        if not root:
            return
        self.traverse(root.left, res)
        self.traverse(root.right, res)
        res.append(root.val)
#105. Construct Binary Tree from Preorder and Inorder Traversal
class Solution:
    def reConstructBinaryTree(self, pre, tin):
        if not pre or not tin:
            return None
        rootVal = pre[0]
        id = tin.index(pre[0])
        root = TreeNode(rootVal)
        root.left = self.reConstructBinaryTree(pre[1:id+1],tin[:id])
        root.right = self.reConstructBinaryTree(pre[id+1:],tin[id+1:])
        return root
""""
四，二叉树的基本变化序
1, DFS遍历，一般不需要返回值。但是可以在遍历的过程中进行各种操作，以达到目的
"""
# 226. Invert Binary Tree
class TreeNode:
    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None
class Solution:
    def invertTree(self,root):
        if not root:
            return None
        self.dfs(root)
        return root
    def dfs(self,root):
        if not root:
            return
        left =root.left
        right = root.right
        root.left = right
        root.right = left
        if root.right:
            self.dfs(root.right)
        if root.left:
            self.dfs(root.left)
""""
2，分治法：分治法一般要返回值，为了求最大值，需要问题转化成解决左子树上深度，
和右子树深度。然后合并中最大的就是全局深度。因为每次遍历深度加一。
"""
#101. Symmetric Tree。Given a binary tree, check whether it is
# a mirror of itself (ie, symmetric around its center。
class TreeNode:
    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None
class Solution:
    def isSymmetric(self, root: TreeNode):
        if not root:
            return True
        return self.checkSymmetric(root.left,root.right)
    def checkSymmetric(self,nodeA,nodeB):
        if nodeA is None or nodeB is None:
            return False
        if nodeA is None and nodeB is None:
            return True
        if nodeA.val != nodeB.val:
            return False
        inner_res = self.checkSymmetric(nodeA.left,nodeB.right)
        outer_res = self.checkSymmetric(nodeA.right, nodeB.left)
        return inner_res and outer_res
""""
3，Path Sum类型问题一般分为三种方式:
1) 判断是否存在从root- leaf 和为指定的数字的路径
（1）构造一个递归出口，一般就是在root 为空时候
（2）构造一个到leaf节点时候 找到指定路径的情况，
    一般就是，当root.left和root.right为空，root.val == 此时target数字
（3）然后进行左右递归，注意，参数中的，root和target数字都是要需要更新的
"""
#112. Path Sum. Given a binary tree and a sum, determine if the tree has a root-to-leaf path such that
# adding up all the values along the path equals the given sum.
class TreeNode:
    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None
class Solution:
    def hasPathSum(self,root,sum):
        if root is None:
            return False
        if root.left is None and root.left is None:
            return sum == root.val
        return self.hasPathSum(root.left, sum-root.val) or self.hasPathSum(root.right, sum-root.val)
#437. Path Sum III. You are given a binary tree in which each node contains an integer value
#Find the number of paths that sum to a given value.The path does not need to start or end at
#the root or a leaf, but it must go downwards (traveling only from parent nodes to child nodes)
#The tree has no more than 1,000 nodes and the values are in the range -1,000,000 to 1,000,000
class TreeNode:
    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None
class Solution:
    def pathSum(self, root: TreeNode, sum: int):
 # 思路分析： 因为题目中要求，只要path之和等于所给的sum，可以不必须是从root开始，而是可以从任何地方开始。
 # 那么就需要考虑，root, root.left, root.right 三种情况下的。总的path的数量就是他们三个所包含的之和
        if not root:
            return 0
        res = []
        self.helper(root, [], sum, res)
        return res
    def helper(self, root, path, target, res):
        if not root:
            return
        path += [root.val]
        flag = (root.left == None and root.right === None)
        if root.val ==target and flag:
            res.append(path[:])
            return
        if root.left:
            self.helper(root.left, path[:], target-root.val,res)
        if root.right:
            self.helper(root.right, path[:], target-root.val,res)
        path.pop()
"""
4，关于最大最小长度问题：一般采用的是分治法
(1) 一般需要返回三类值:1)curLen/curSum, 返回当前所求的长度和值；2）maxLen/maxSum，返回最值; 3) node或者nodeVal,
返回节点或者节点数值，具体选择哪一个，看题中要求；
(2) 对于空集的判断if not root,需要注意的是，最值在初始化的时候，最大值一般是用-sys.maxsizem,最小值是用sys.maxsize
(3) divide，分别对左右进行循环
(4) conque, 在conque的时候，需要分两步:1)先根据调解，求出curLen/curSum；2）然后在判断出最值
"""
# Minimum Subtree.Given a binary tree, find the subtree with minimum sum. Return the root of the subtree
class TreeNode:
    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None
class Solution:
    def findSubtree(self, root: TreeNode):
        if not root:
            return None
        node, minSubtree,Sum = self.helper(root)
        return node
    def helper(self,root):
        if not root:
            return None, 0, sys.maxsize
        left_node, left_min_sum, left_sum = self.helper(root.left)
        right_node, right_min_sum, right_sum = self.helper(root.right)
        total_sum = left_sum + root.val+ right_sum
        if left_min_sum == min(left_min_sum, right_min_sum, total_sum):
            return left_node, left_min_sum, total_sum
        if right_min_sum == min(left_min_sum, right_min_sum, total_sum):
            return right_node , right_min_sum, total_sum
        if total_sum == min(left_min_sum, right_min_sum, total_sum):
            return root, total_sum, total_sum
# 298. Binary Tree Longest Consecutive Sequence.Given a binary tree, find the length of the longest consecutive sequence path
# The path refers to any sequence of nodes from some starting node to any node in the tree along the parent-child connections.
# The longest consecutive path need to be from parent to child (cannot be the reverse)
class TreeNode:
    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None
class Solution:
    def longestConsecutive(self, root: TreeNode):
        if not root:
            return 0
        maxLen, curLen, curVal = self.helper(root)
        return maxLen
    def helper(self, root):
        if not root:
            return -sys.maxsize, 0, 0
        left_maxLen, left_curLen, left_curVal = self.helper(root.left)
        right_maxLen, right_curLen, right_curVal = self.helper(root.right)
        curLen =1
        if root.val == left_curVal -1:
            curLen = max(curLen, left_curLen+1)
        if root.val == right_curVal -1:
            curLen = max(curLen, right_curLen+1)
        if left_maxLen == max(curLen, left_maxLen, right_maxLen):
            return left_maxLen, curLen, root.val
        if right_maxLen == max(curLen, left_maxLen, right_maxLen):
            return right_maxLen, curLen, root.val
        return curLen,curLen,root.val
"""
5, 二叉树宽度有限搜索BFS：主要是使用deque这类函数先进先出的优点，配合deque.popleft(), 实现层序遍历
"""
# 102. Binary Tree Level Order Traversal。Given a binary tree, return the level order traversal
# of its nodes' values. (ie, from left to right, level by level)
class TreeNode:
    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None
class Solution:
    def levelOrder(self, root: TreeNode):
        if not root:
            return []
        queue = deque([root])
        res = []
        while queue:
            level = []
            for _ in range(len(queue)):
                node = queue.popleft()
                level.append(node.val)
                if node.left:
                    queue.append(node.left)
                if node.right:
                    queue.append(node.right)
            res.append(level)
        return res
#107, Binary Tree Level Order Traversal II.Given a binary tree, return the bottom-up level order traversal of its nodes' values.
# (ie, from left to right, level by level from leaf to root)
class TreeNode:
    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None
from collections import deque
class Solution:
    def levelOrderBottom(self, root: TreeNode):
        if not root:
            return []
        queue = deque([root])
        res = []
        while queue:
            level = []
            for _ in range(len(queue)):
                node = queue.popleft()
                level.append(node.val)
                if node.left:
                    queue.append(node.left)
                if node.right:
                    queue.append(node.right)
            res.insert(0,level)
        return res
#103,Binary Tree Zigzag Level Order Traversal.
#Given a binary tree, return the zigzag level order traversal of its nodes' values. (ie, from left to right,
# then right to left for the next level and alternate between)."
class TreeNode:
    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None
from collections import deque
class Solution:
    def zigzagLevelOrder(self, root: TreeNode) -> List[List[int]]:
        if not root:
            return []
        queue = deque([root])
        res = []
        while queue:
            count +=1
            level = []
            for _ in range(len(queue)):
                node = queue.popleft()
                if count%2 ==1:
                    level.append(node.val)
                else:
                    level.insert(0,node.val)
                if node.left:
                    queue.append(node.left)
                if node.right:
                    queue.append(node.right)
            res.append(level)
        return res
"""
6,二叉搜索树(binary search tree)相关问题.二叉搜索树的条件：1）左子树值小于根节点； 2）根节点值小于右子树； 3）左右子树都是二叉搜索树BST
"""
#426. Convert Binary Search Tree to Sorted Doubly Linked List.Convert a BST to a sorted circular doubly-linked list in-place.
# Think of the left and right pointers as synonymous to the previous and next pointers in a doubly-linked list.For a circular
# doubly linked list, the predecessor of the first element is the last element, and the successor of the last element is the first element.
class TreeNode:
    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None
from collections import deque
class Solution:
    def treeToDoublyList(self, root: TreeNode) -> List[List[int]]:
        if not root:
            return []
        res = []
        array = self.inOrderTraversal(root,res)
        head = self.constructDoublyLinkedList(array)
        return head
    def inOrderTraversal(self,root,res): # left root right
        if not root:
            return []
        res = []
        stack =[]
        while root:
            stack.append(root)
            root = root.left
        while stack:
            node = stack.pop()
            res.append(node)
            if node.right:
                node = node.right
                while node:
                    stack.append(node)
                    node = node.left
        return res
    def constructDoublyLinkedList(self, array):
        array[-1].right = array[0]
        array[0].left = array[-1]
        for i in range(len(array)):
            array[i-1].right = array[i]
            array[i].left = array[i-1]
        return array[0]
# 285. Inorder Successor in BST. Given a binary search tree and a node in it, find the in-order successor of that node in the BST
#The successor of a node p is the node with the smallest key greater than p.val
# 中序后继: 二叉树中序是 left root right, 二叉搜索树，左子树小于根节点，根节点小于右子树，左右子树都是二叉树。
# 二叉搜索树的中序遍历是个递增的数组，顺序后继是中序遍历中当前节点 之后最小的节点
# 可以分为两种情况来讨论：1）如果当前节点有右孩子，顺序后继在当前节点之下；如果当前节点没有右孩子，顺序后继在当前节点之上。
# 算法逻辑： 1）如果当前节点有右子树，找到右子树，再持续往左直到节点左孩子为空，直接返回该节点；2）如果当前节点没有右子树，就需要使用非递归
# 的中序遍历，维持一个栈，当栈中有节点时：a,往左走直到节点的左孩子为空，并将每个访问的节点压入栈中；b,弹出栈中节点，判断当前的前继节点是否为p
# 如果是则直接返回当前节点，如果不是，将当前节点赋值给前继节点；c, 往右走一步；3）如果走到这一步，说明不存在后续，返回空
class Solution:
    def inorderSuccessor(self, root: 'TreeNode', p: 'TreeNode') -> 'TreeNode':
        if p.right:
            p = p.right
            while p.left:
                p = p.left
            return p
        stack = []
        inorder_precessor_val = float('-inf') # set a value as the precessor's
        # if p have no right node, the inordersuccessor must be somewhere as the right node of it's smallest left nodes
        while root or stack:
            # go to left till the smallest left node
            while root:
                stack.append(root)
                root = root.left
            root = stack.pop()  # the smallest left node
            if inorder_precessor_val == p.val:
                return root
            inorder_precessor_val = root.val
            root = root.right
        return None
# 时间复杂度： 如果节点p有右孩子，时间复杂度为O(H_p）其中H_p是节点p的高度。如果没有右孩子，时间复杂度为O(H)，其中H为树的高度
# 空间复杂度： 如果节点p有右孩子，空间复杂度为O(1).如果没有右孩子，空间复杂度为O(H)
#1008.Construct Binary Search Tree from Preorder Traversal. Given an array of integers
class Solution:
    def bstFromPreorder(self, preorder: List[int]) -> TreeNode:
        root_val = preorder[0]
        root = TreeNode(root_val)
        stack = []
        stack.append(root)
        for value in preorder[1:]:
            if stack[-1].val>value:
                stack[-1].left = TreeNode(value)
                stack.append(stack[-1].left)
            else:
                while stack and stack[-1].val<value:
                    last = stack.pop()
                last.right = TreeNode(value)
                stack.append(last.right)
        return root
# 230. Kth Smallest Element in a BST. Given a binary search tree, write a function kthSamallest to find the
# kth smallest element in it. You may assume k is always valid, 1 ≤ k ≤ BST's total elements.
# 搜索二叉树的Inorder Traversal 所得到的就是递增数列
class Solution:
    def kthSmallest(self, root: TreeNode, k: int) -> int:
        if not root:
            return None
        in_order_traversal = self.inOrderTraveral(root)
        return in_order_traversal[k - 1]
    def inOrderTraveral(self, root):  # left root right
        if not root:
            return []
        res = []
        stack = []
        while root:
            stack.append(root)
            root = root.left
        while stack:
            node = stack.pop()
            res.append(node.val)
            if node.right:
                node1 = node.right
                while node1:
                    stack.append(node1)
                    node1 = node1.left
        return res
# 236. Lowest Common Ancestor of a Binary Tree.Given a binary tree, find the lowest common ancestor (LCA)
# of two given nodes in the tree. According to the definition of LCA on Wikipedia: “The lowest common
# ancestor is defined between two nodes p and q as the lowest node in T that has both p and q as descendants
# (where we allow a node to be a descendant of itself)
class Solution(object):
    def lowestCommonAncestor(self, root, p, q):
        if not root:
            return None
        if root == p or root == q:
            return root
        left_result = self.lowestCommonAncestor(root.left, p, q)
        right_result = self.lowestCommonAncestor(root.right, p, q)
        if left_result and right_result:
            return root
        if left_result:
            return left_result
        if right_result:
            return right_result
        return None
"""
DFS(Deep First Search)
--Definition: 一条路走到底，直到无路可走时回退到上一步。Search通过遍历的方式达到搜索的目的。
1）树的深度优先搜索：前中后序遍历都是深度优先遍历，其中后序遍历的思想在解决一些树的问题中非常有用，，需在做题中不断体会和总结
2）图的深度优先搜索：由于图中存在"环"，须要将记录已经遍历过的节点标记为"已经访问"
--整体描述：
DFS只要前面有可以走的路，就会一直向前走，直到无路可走才回头，无路可走分为两个情况：（1)遇到了墙；(2)遇到了已经走过的路。在无路可走的时候，沿着原路返回，直到回到了还未走过的路口，
尝试继续走没有走过的路径；有一些路径没有走到，就是因为找到了出口，程序就停止了。
"""
""""
TX的top 10算法
1， CSIG： 
补充题4. 手撕快速排序；206. 反转链表；704. 二分查找； 415. 字符串相加； 补充题6. 手撕堆排序；102. 二叉树的层序遍历；
470. 用 Rand7() 实现 Rand10()； 53. 最大子序和； 4. 寻找两个正序数组的中位数； 141. 环形链表	
2，IEG：
146. LRU缓存机制；160. 相交链表； 1. 两数之和；155. 最小栈；
232. 用栈实现队列；21. 合并两个有序链表；25. K 个一组翻转链表；
3，PCG：
215. 数组中的第K个最大元素；
5. 最长回文子串；70. 爬楼梯；15. 三数之和；补充题23. 检测循环依赖；
4，TEG：
121. 买卖股票的最佳时机；
补充题22. IP地址与整数的转换；236. 二叉树的最近公共祖先；
5，CDG：
；227. 基本计算器 II；143. 重排链表；20. 有效的括号；144. 二叉树的前序遍历；460. LFU缓存； 
3. 无重复字符的最长子串；118. 杨辉三角； 31. 下一个排列；
6，WXG：
8. 字符串转换整数 (atoi)；153. 寻找旋转排序数组中的最小值；
剑指 Offer 54. 二叉搜索树的第k大节点； 300. 最长上升子序列；2. 两数相加；112. 路径总和
"""
# 补充题4. 手撕快速排序
class Solution:
    def sortArray(self, nums: List[int]) -> List[int]:
        if not nums:
            return []
        self.quickSort(nums, 0, len(nums)-1)
        return nums
    def quickSort(self,nums,start, end):
        if start >=end:
            return
        left,right = start, end
        pivot = (nums[left]+nums[right])/2
        while left <= right :
            while left <= right and nums[left]<pivot:
                left+=1
            while left>=right and nums[right]>pivot:
                right-=1
            if left<=right:
                nums[left], nuums[right] = nums[right], nums[left]
                left+=1
                right-=1
        self.quickSort(nums, start, right)
        self.quickSort(nums, left, end)
# 206. 反转链表
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution:
    def reverseList(self, head: ListNode) -> ListNode:
        # 可以理解为双指针，维护两个指针，一个是pre,一个是cur
        cur = head
        pre = None
        while cur:
            # 先记录下当前节点的下个节点
            tmp = cur.next
            # 让当前节点反向志向pre
            cur.next = pre
            # 此时pre在cur的左边，cur赋予给pre，让pre向前走一步
            pre = cur
            # 把cur.next给cur，向前走一步
            cur = tmp
        # 翻转结束之后，新链表的头就是pre, 尾部就是head(head在这个过程中没有发生变化)
        return pre
# 704. 二分查找
#Given an array of integers nums which is sorted in ascending order, and an integer target, write a function to search target in nums.
#If target exists, then return its index. Otherwise, return -1.You must write an algorithm with O(log n) runtime complexity.
class Solution:
    def search(self, nums: List[int], target: int) -> int:
        if not nums or target is None:
            return -1
        start, end = 0, len(nums)-1
        while start<=end:
            mid = (start+end)//2
            if nums[mid] == target:
                return mid
            elif nums[mid] > target:
                end = mid -1
            else:
                start = mid+1
        return -1
#415. 字符串相加，
#Given two non-negative integers, num1 and num2 represented as string, return the sum of num1 and num2 as a string.
#You must solve the problem without using any built-in library for handling large integers (such as BigInteger).
#You must also not convert the inputs to integers directly.
class Solution:
    def addStrings(self, num1: str, num2: str) -> str:
        res = ""
        i,j,carry = len(num1)-1, len(num2)-1,0
        while i>=0 or j>=0:
            n1 =int(num1[i]) if i>=0 else 0
            n2 = int(num2[j]) if j>=0 else 0
            tmp = n1+n2+carry
            carry = tmp//10
            res = str(tmp%10)+res
            i,j=i-1,j-1
        return "1"+res if carry else res
#补充题6. 手撕堆排序
#堆就是一个完全二叉树，完全二叉树的定义：深度为h的完全二叉树，除了第h层外，其他各层(1~h-1)的节点数都达到最大，第h层所有的节点都集中在最左边。
#如果堆总共有n个节点，那么堆的最后一个非叶子节点是第n//2-1
def heapSort(arr):
    build_max_heap(arr)
    lenth = len(arr)
    for i in range(lenth-1, 0, -1):
        swap(arr,0, i)
        max_heapify(arr,0,i)
def build_max_heap(arr):
    n = len(arr)
    for i in range(n/2-1, 0, -1):
        max_heapify(arr,i,n)
def max_heapify(arr,i,heap_size):
    l = 2*i+1
    r = l+1
    largest = i
    if heap_size>1 and arr[l]>arr[largest]:
        largest = l
    if heap_size>1 and arr[r]>arr[largest]:
        largest = l
    if largest != i:
        swap(arr,i,largest)
        max_heapify(arr, largest, heap_size)
def swap(arr, i, j):
    temp = arr[i]
    arr[i] = arr[j]
    arr[j] = temp
# 102. 二叉树的层序遍历
# 102. Binary Tree Level Order Traversal
# Given a binary tree, return the level order traversal of its nodes' values. (ie, from left to right, level by level
class TreeNode:
    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None
from collections import deque
class Solution:
    def levelOrder(self,root:TreeNode) -> List[List[int]]:
        if not root:
            return []
        queue = deque([root])
        res = []
        while queue:
            level = []
            for _ in range(len(queue)):
                node = queue.popleft()
                level.append(node.val)
                if node.left:
                    queue.append(node.left)
                if node.right:
                    queue.append(node.right)
                res.append(level)
        return res
# 470. 用 Rand7() 实现 Rand10()
# Given the API rand7() that generates a uniform random integer in the range [1, 7], write a function rand10() that generates a uniform random integer in the range [1, 10].
# You can only call the API rand7(), and you shouldn't call any other API. Please do not use a language's built-in random API. Each test case will have one internal argument n,
# the number of times that your implemented function rand10() will be called while testing. Note that this is not an argument passed to rand10().
class Solution:
    def rand10(self):
        # 先在0-5之间均匀分布，去掉6，7，然后再加一个0或者5，就可以映射到0-10之间
        n = rand7()
        while n >5: #使用while的目的，是满足均匀概率分布
            n = rand7()
        i = rand7()
        while i ==4:
            i = rand7()
        if i > 5:
            j = 0
        else:
            j = 5
        return n+j
# 53. 最大子序和
#Given an integer array nums, find the contiguous subarray (containing at least one number) which has the largest sum and return its sum.
# A subarray is a contiguous part of an array.
class Solution:
    def maxSubArray(self,nums):
        if not nums or len(nums)==0:
            return 0
        n = len(nums)
        f = n*[0]
        f[0]=nums[0]
        for i in range(1,n):
            f[i] = nums[i]+max(f[i-1],0)
        max_value = -sys.maxsize -1
        for v in f:
            max_value = max(v, max_value)
        return max_value
# 4. 寻找两个正序数组的中位数
# Given two sorted arrays nums1 and nums2 of size m and n respectively, return the median of the two sorted arrays.
# The overall run time complexity should be O(log (m+n)).
# 总体思路：要求得到的中位数，如果两个数组直接合并到一起，使用merge_sort的方式，复杂度是O(m+n)，不符合要求，但是可以合并到一起分析出来，如果合并的数组
# 是偶数，得到的结果就是中间两个数的中位数；如果是奇数，就是中间的那个数。
# 因为是log的复杂度，所以一定是二分法，因为两个序列都是sorted的，那么可以把两个数组都拆分成左右两个部分，中间的那个数便是要找的数。
# nums1被拆开为L1,R1；nums2被拆开为L2,R2。那么最终找到的中位数，其左边的数一定是L1+L2（为两个数列数量之和的一半）；然后还需要满足的条件：
# L1<R1&&L1<R2 L2<R1&&L2<R2
class Solution:
    def findMedianSortedArrays(self, nums1: List[int], nums2: List[int]) -> float:
        # choose the nums with smaller length to implement sort as the time complexity would be O(log(min(m,n)))
        n1, n2 = len(nums1), len(nums2)
        total_len = n1 + n2
        if n1 > n2:
            return self.findMedianSortedArrays(nums2, nums1)
        if n1 == 0:
            if n2 % 2!= 0:
                return nums2[total_len // 2]
            else:
                return nums2[total_len // 2 - 1] + nums2[total_len // 2]
        left_edge, right_edge = 0, n1
        # cur1, cur2 represent the position in nums1, nums2
        result = 0
        while left_edge <= right_edge:
            cur1 = (left_edge + right_edge) // 2
            cur2 = (total_len + 1) // 2 - cur1
            # figure out the L1,R1,L2,R2
            if cur1 == 0:
                L1 = -sys.maxsize
                R1 = nums1[cur1]
            elif cur1 == n1:
                L1 = nums1[cur1 - 1]
                R1 = sys.maxsize
            else:
                L1 = nums1[cur1 - 1]
                R1 = nums1[cur1]
            if cur2 == 0:
                L2 = -sys.maxsize
                R2 = nums2[cur2]
            elif cur2 == n2:
                L2 = nums2[cur2 - 1]
                R2 = sys.maxsize
            else:
                L2 = nums2[cur2 - 1]
                R2 = nums2[cur2]
            # Binary search, find the boundary
            if L1 > R2:
                right_edge = cur1 - 1
            elif L2 > R1:
                left_edge = cur1 + 1
            else:
                # if the length is odd,choose the middle one. if even, choose the half of two middles' sum
                if total_len % 2 != 0:
                    result = max(L1, L2)
                else:
                    result = (max(L1, L2) + min(R1, R2)) / 2
                break
        return result
# 141. 环形链表 Given a linked list, determine if it has a cycle in it
class ListNode:
    def __init__(self, x):
        self.val = x
        self.next = None
class Solution:
    def hasCycle(self, head: ListNode) -> bool:
        if not head or not head.next:
            return False
        p1 = head
        p2 = head
        while p2 and p2.next:
            p1 = p1.next
            p2 = p2.next.next
            if p1 is p2:
                return True
        return False
# 146. LRU缓存机制；Design a data structure that follows the constraints of a Least Recently Used (LRU) cache.
# Implement the LRUCache class:
# LRUCache(int capacity) Initialize the LRU cache with positive size capacity.
# int get(int key) Return the value of the key if the key exists, otherwise return -1.
# void put(int key, int value) Update the value of the key if the key exists. Otherwise, add the key-value pair to the cache.
# If the number of keys exceeds the capacity from this operation, evict the least recently used key.
# The functions get and put must each run in O(1) average time complexity.
# 时间复杂度： 对于put和get都是O(1)
# 空间复杂度：O(capacity),因为哈希表和双向链表最多可以存储capacity+1个元素
class DLinkedNode:
    def __init__(self, key=0, value=0):
        self.key = key
        self.value = value
        self.prev = None
        self.next = None
class LRUCache:
    def __init__(self, capacity:int):
        self.cache = dict()
        # use dummy head and dummy tail
        self.head = DLinkedNode()
        self.tail = DLinkedNode()
        self.head.next = self.tail
        self.tail.prev = self.head
        self.capacity = capacity
        self.size = 0
    def get(self, key:int):
        if key not in self.cache:
            return -1
        # if key exist, locating the key by hash and then move to the head of dlinked'
        node = self.cache[key]
        self.moveToHead(node)
        return node.value
    def put(self, key, value):
        if key not in self.cache:
            # if key not exist, create a new node
            node = DLinkedNode(key, value)
            # add to hash
            self.cache[key] = node
            # add to the head of dlinked list
            self.addToHead(node)
            self.size+=1
            if self.size > self.capacity:
                # if larger than capacity, delete the tail node of dlinked
                removed = self.removeTail()
                self.cache.pop(removed.key)
                self.size -=1
        else:
            # if key exist, use hash to locate,then modify the value and move to the head
            node = self.cache[key]
            node.value = value
            self.moveToHead(node)
    def addToHead(self,node):
        node.prev = self.head
        node.next = self.head.next
        # remind this is double linked list
        self.head.next.prev = node
        self.head.next = node
    def removeNode(self,node):
        node.prev.next = node.next
        node.next.prev = node.prev
    def moveToHead(self, node):
        self.removeNode(node)
        self.addToHead(node)
    def removeTail(self):
        node = self.tail.prev
        self.removeNode(node)
        return node
#160. 相交链表：Given the heads of two singly linked-lists headA and headB, return the node at which the two lists intersect.
# If the two linked lists have no intersection at all, return null.
# Total length of Linkedlist A,B is a,b. The common length of intersection is c. Assign two pointers pA and pB move at the same times.
# Only a = b, can A meet B at the first round. In the second round, assign pA to head B, so the distance for A before it to the intersection
# would be a+b-c. assign pB to headA, so the distance for B before it would be b+a-c. The distance of A equal B
class ListNode:
    def __init__(self, x):
        self.val = x
        self.next = None
class Solution:
    def getIntersectionNode(self, headA: ListNode, headB: ListNode) -> ListNode:
        if not headA or not headB:
            return None
        pA, pB = headA, headB
        while pA!=pB:
            if pA:
                pA = pA.next
            else:
                pA = headB
            if pB:
                pB = pB.next
            else:
                pB = headA
        return pA
# 1. 两数之和。Given an array of integers nums and an integer target, return indices of the two numbers such that they add up to target.
# You may assume that each input would have exactly one solution, and you may not use the same element twice.
# You can return the answer in any order.
class Solution:
    # use hashmap, time complexity O(n), space complexity O(n)
    def twoSum(self, nums: List[int], target: int) -> List[int]:
        if not nums:
            return []
        hashmap = {}
        for i in range(0, len(nums)):
            if target - nums[i] in hashmap:
                return [hashmap[target - nums[i]], i]
            hashmap[nums[i]] = i
        return -1
    # use two pointer, time complexity O(nlogn): sorted time complexity is O(nlogn), two pointer find target is O(n), space complexity O(n)
    def twoSum(self, nums: List[int], target: int) -> List[int]:
        if not nums:
            return []
        tmp = sorted(nums)
        left,right = 0, len(nums)-1
        while left<right:
            if tmp[left]+tmp[right] == target:
                i, j = nums.index(tmp[left]),nums.index(tmp[right])
                if i==j:
                    index = nums[i+1:].index(tmp[right])+i+1
                    return [i, index]
                elif i>j:
                    return [j, i]
                else:
                    return [i, j]
            elif tmp[left]+tmp[right] >target:
                right-=1
            else:
                left+=1
        return -1
# 155. 最小栈
# Design a stack that supports push, pop, top, and retrieving the minimum element in constant time.
# Implement the MinStack class:
# MinStack() initializes the stack object.
# void push(val) pushes the element val onto the stack.
# void pop() removes the element on the top of the stack.
# int top() gets the top element of the stack.
# int getMin() retrieves the minimum element in the stack.
# Time complexity: O(1)，all operations remove/read/push are O(1); Space complexity:O(n)
class MinStack:
    def __init__(self):
        self.stack = []
        self.min_stack = [math.inf]
    def push(self, x):
        self.stack.append(x)
        self.min_stack.append(min(x,self.min_stack[-1]))
    def pop(self):
        self.stack.pop()
        self.min_stack.pop()
    def top(self):
        self.stack[-1]
    def getMin(self):
        return self.min_stack[-1]
# 232. 用栈实现队列
    def __init__(self):
        """
        Initialize your data structure here.
        """
        self.s1 = []
        self.s2 = []
        self.front = None
    def push(self, x: int) -> None:
        """
        Push element x to the back of queue.
        """
        if not self.s1: self.front = x
        self.s1.append(x)
    def pop(self) -> int:
        """
        Removes the element from in front of queue and returns that element.
        """
        if not self.s2:
            while self.s1:
                self.s2.append(self.s1.pop())
            self.front = None
        return self.s2.pop()
    def peek(self) -> int:
        """
        Get the front element.
        """
        if self.s2:
            return self.s2[-1]
        return self.front
    def empty(self) -> bool:
        """
        Returns whether the queue is empty.
        """
        if not self.s1 and not self.s2:
            return True
        return False
# 21. 合并两个有序链表
class ListNode:
    def __init__(self, val = 0, next = None):
        self.val = val
        self.next = next
class Solution:
    def merge_two_lists(self, l1:ListNode, l2:ListNode):
        if not l1:
            return l2
        if not l2:
            return l1
        dummy = ListNode(0)
        tmp = dummy
        tmp1 = l1
        tmp2 = l2
        while tmp1 and tmp2:
            if tmp1.val<tmp2.val:
                tmp.next = tmp1
                tmp1 = tmp1.next
            else:
                tmp.next = tmp2
                tmp2 = tmp2.next
            tmp = tmp.next
        if tmp1:
            tmp.next = tmp1
        if tmp2:
            tmp.next = tmp2
        return dummy.next
# 25，Reverse Nodes in k-Group
# Given a linked list, reverse the nodes of a linked list k at a time and return its modified list.
# #k is a positive integer and is less than or equal to the length of the linked list.
# If the number of nodes is not a multiple of k then left-out nodes, in the end, should remain as it is.
# You may not alter the values in the list's nodes, only nodes themselves may be changed.
class ListNode:
    def __init__(self, val = val, next = None):
        self.val = val
        self.next = None
class Solution:
    class Solution:
        def reverseKGroup(self, head: ListNode, k: int) -> ListNode:
            hair = ListNode(0)
            hair.next = head
            pre = hair
            while head:
                tail = pre
                # 查看剩余部分长度是否大于等于 k
                for i in range(k):
                    tail = tail.next
                    if not tail:
                        return hair.next
                nex = tail.next
                head, tail = self.reverse(head, tail)
                # 把子链表重新接回原链表
                pre.next = head
                tail.next = nex
                pre = tail
                head = tail.next
            return hair.next
        def reverse(self, head: ListNode, tail: ListNode):
            prev = tail.next
            p = head
            while prev != tail:
                nex = p.next
                p.next = prev
                prev = p
                p = nex
            return tail, head
# 215. Kth Largest Element in an Array
# 第一步：先partition，
class Solution:
    def findKthLargest(self, nums: List[int], k: int) -> int:
        if not nums or k:
            return
        self.quickSelect(nums,0,len(nums)-1,k)
        return nums[k-1]
    def quickSelect(self,nums,start,end,k):
        if not nums or k:
            return
        left, right = start, end
        pivot = (nums[left] + nums[right]) // 2
        while left<=right:
            while left<=right and nums[left]<pivot:
                left+=1
            while left<=right and nums[right]>pivot:
                right-=1
            if left<=right:
                nums[left], nums[right] = nums[right], nums[left]
                left+=1
                right-=1
        if start+k-1<=right:
            self.quickSelect(nums,start,right,k)
        if start+k-1>=left:
            self.quickSelect(nums,left, end, k-(left-start))
#5, Longest Palindromic Substring：Given a string s, return the longest palindromic substring in s.
class Solution:
"""方法一，动态规划：时间复杂度O(n^2), 空间复杂度O(n)"""
    def longestPalindrome(self, s: str) -> str:
        if not s:
            return 0
        n = len(s)
        f = [[False]*n for _ in range(n)]
        f[0][0] = True
        for i in range(n):
            f[i][i] = True
        for i in range(n):
            if s[i] == s[i-1]:
                f[i][i-1] = True
        maxLen, start, end = 1,0,0
        for lenth in range(1,n):
            for i in range(n-lenth):
                j = i+lenth
                f[i][j] = (s[i]==s[j] and f[i+1][j-1])
                if f[i][j] is True and maxLen<lenth+1:
                    maxLen = lenth+1
                    start, end = i, j
        return s[start:end+1]
"""方法二，中心扩展法：时间复杂度O(n^2), 空间复杂度O(1)"""
    def longestPalindrome(self, s: str) -> str:
        # 从0开始去计算
        start, end =0,0
        for i in range(n):
            left1, right1 = self.expand_from_center(s,i,i)
            left2, right2 = self.expand_from_center(s, i, i+1)
            if right1-left1>end-start:
                start, end = left1, right1
            if right2-left2>end-start:
                start, end = left2, right2
        return s[start:end+1]
def expand_from_center(self,s,left,right):
    # 从中心扩展寻找是回文串最大窗口
        while left>=0 and right<len(s) and s[left]==s[right]:
            left-=1
            right+=1
        return left+1, right-1
# 70. Climbing Stairs.You are climbing a staircase. It takes n steps to reach the top.
# Each time you can either climb 1 or 2 steps. In how many distinct ways can you climb to the top?
class Solution:
    def climbStairs(self, n: int) -> int:
        if n==0:
            return 0
        f=[0]*(n+1)
        for i in range(1,n+1):
            if i<=2:
                f[i]=i
            else:
                f[i] = f[i-1]+f[i-2]
        return f[-1]
#15. 3Sum, Given an integer array nums, return all the triplets [nums[i], nums[j], nums[k]] such that i != j, i != k, and j != k, and nums[i] + nums[j] + nums[k] == 0.
# Notice that the solution set must not contain duplicate triplets.
"""双指针加排序法： 时间复杂度O(n^2), 空间复杂度O(1)"""
class Solution:
    def threeSum(self, nums: List[int]) -> List[List[int]]:
        if not nums or len(nums)<3:
            return []
        nums.sort()
        res =[]
        n = len(nums)
        for k in range(n-2):
            if k and nums[k] == nums[k-1]:
                continue
            i, j = k+1, n-1
            target = -nums[k]
            while i<j:
                if nums[i]+nums[j] == target:
                    res.append([nums[k],nums[i],nums[j]])
                    i+=1
                    j-=1
                    while i<j and nums[i] == nums[i-1]:
                        i+=1
                    while i<j and nums[j] == nums[j+1]:
                        j-=1
                elif nums[i]+nuums[j]>target:
                    j-=1
                else:
                    i+=1
            return res
# 剑指 Offer II 022. 链表中环的入口节点,给定一个链表，返回链表开始入环的第一个节点。 从链表的头节点开始沿着 next 指针进入环的第一个节点为环的入口节点。如果链表无环，则返回 null。
# 为了表示给定链表中的环，我们使用整数 pos 来表示链表尾连接到链表中的位置（索引从 0 开始）。 如果 pos 是 -1，则在该链表中没有环。注意，pos 仅仅是用于标识环的情况，并不会作为参数传递到函数中。
"""双指针，时间复杂度O(n),空间复杂度O(n)
use two pointer p1, p2; p1 = p1.next, p2 =p2.next.next 
l1: the length of cycle entry from head
l2: cycle length 
a: length walked for p1 from the entry of cycle when first p1 met p2
when first p1 met p2, length for p1 = l1+a, p2 = l1+l2+a  --> 2*l1+2a = l1+l2+a -->l1+a = l2 --> l1=l2-a 
therefor, when first met, put p2 to the head and walk one step, when p1 met p2 again, it would be the cycle entry
"""
class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next
class Solution:
    def detectCycle(self, head: ListNode) -> ListNode:
        if not head or not head.next:
            return None
        p1, p2 = head, head
        while p2 and p2.next:
            p1 = p1.next
            p2 = p2.next.next
            if p1 is p2:
                break
        if p1 is p2:
            p2 = head
            while p1 is not p2:
                p1 = p1.next
                p2 = p2.next
            return p2
        return None
# 121. Best Time to Buy and Sell Stock,You are given an array prices where prices[i] is the price of a given stock on the ith day.
# You want to maximize your profit by choosing a single day to buy one stock and choosing a different day in the future to sell that stock.
# Return the maximum profit you can achieve from this transaction. If you cannot achieve any profit, return 0.
class Solution:
    def max_profit(self, prices):
        max_profit = 0
        min_price = sys.maxsize
        for price in prices:
            min_price = min(min_price, price)
            max_profit = max(max_profit, price-min_price)
        return max_profit
# 122. Best Time to Buy and Sell Stock II;
# You are given an integer array prices where prices[i] is the price of a given stock on the ith day.
# On each day, you may decide to buy and/or sell the stock. You can only hold at most one share of the stock at any time. However, you can buy it then immediately sell it on the same day.
# Find and return the maximum profit you can achieve.
"""因为每天都可以及时卖出和购买，所以只要当天价格比前一天高，就卖出，最终累计的一定是最大利益，贪心法，时间复杂度O(n),空间复杂度O(1)"""
class Solution:
    def max_profit(self, prices):
        profit = 0
        for i in range(1,len(prices)):
            if prices[i]>prices[i-1]:
                profit += prices[i] - prices[i-1]
        return profit
# 309. Best Time to Buy and Sell Stock with Cooldown
# You are given an array prices where prices[i] is the price of a given stock on the ith day.
# Find the maximum profit you can achieve. You may complete as many transactions as you like (i.e., buy one and sell one share of the stock multiple times) with the following restrictions:
# After you sell your stock, you cannot buy stock on the next day (i.e., cooldown one day).
# Note: You may not engage in multiple transactions simultaneously (i.e., you must sell the stock before you buy again).
"""
交易分为三个阶段，sell, cooldown and buy. 因此构造三个动态规划矩阵sell =[0]*n, cooldown =[0]*n and buy =[0]*n 分别表示在i的时候，sell/cooldown
/buy时候的最大收益状态
"""
class Solution:
    def maxProfit(self, prices: List[int]) -> int:
        n = len(prices)
        sell = [0]*n
        buy = [0]*n
        cooldown = [0]*n
        buy[0] = -prices[0]
        for i in range(1,n):
            cooldown[i] = sell[i-1]
            sell[i] = max(sell[i-1], prices[i]+buy[i-1])
            buy[i] = max(buy[i-1], cooldown[i-1]-prices[i])
        return max(sell[n-1], cooldown[n-1])
# 123. Best Time to Buy and Sell Stock III
# You are given an array prices where prices[i] is the price of a given stock on the ith day.
# Find the maximum profit you can achieve. You may complete at most two transactions.
# Note: You may not engage in multiple transactions simultaneously (i.e., you must sell the stock before you buy again).
class Solution:
    def maxProfit(self, prices: List[int]) -> int:
        n = len(prices)
        buy1 = buy2 = -prices[0]
        sell1 = sell2 = 0
        for i in range(1, n):
            buy1 = max(buy1, -prices[i])
            sell1 = max(sell1, buy1 + prices[i])
            buy2 = max(buy2, sell1 - prices[i])
            sell2 = max(sell2, buy2 + prices[i])
        return sell2
# 236. Lowest Common Ancestor of a Binary Tree
#Given a binary tree, find the lowest common ancestor (LCA) of two given nodes in the tree.
#According to the definition of LCA on Wikipedia: “The lowest common ancestor is defined between two nodes p and q as the lowest node in T that
#has both p and q as descendants (where we allow a node to be a descendant of itself).”
class TreeNode:
    def __init__(self,x):
        self.val = x
        self.left = None
        self.right = None
class Solutions:
    def lowestCommonAncestor(self, root, p, q):
        if not root:
            return None
        if root ==q or root ==p:
            return root
        right_res = self.lowestCommonAncestor(root.right,p,q)
        left_res = self.lowestCommonAncestor(root.left,p,q)
        if right_res and left_res:
            return root
        if right_res:
            return right_res
        if left_res:
            return left_res
        return None
# 剑指 Offer 67. 把字符串转换成整数
# 写一个函数 StrToInt，实现把字符串转换成整数这个功能。不能使用 atoi 或者其他类似的库函数。
# 首先，该函数会根据需要丢弃无用的开头空格字符，直到寻找到第一个非空格的字符为止。
# 当我们寻找到的第一个非空字符为正或者负号时，则将该符号与之后面尽可能多的连续数字组合起来，作为该整数的正负号；假如第一个非空字符是数字，则直接将其与之后连续的数字字符组合起来，形成整数。
# 该字符串除了有效的整数部分之后也可能会存在多余的字符，这些字符可以被忽略，它们对于函数不应该造成影响。
# 注意：假如该字符串中的第一个非空格字符不是一个有效整数字符、字符串为空或字符串仅包含空白字符时，则你的函数不需要进行转换。
# 在任何情况下，若函数不能进行有效的转换时，请返回 0。
# 说明：
# 假设我们的环境只能存储 32 位大小的有符号整数，那么其数值范围为 [−231,  231 − 1]。如果数值超过这个范围，请返回  INT_MAX (231 − 1) 或 INT_MIN (−231) 。
class Solution:
    def strToInt(self, str: str) -> int:
        res, i, sign, length = 0, 0, 1, len(str)
        int_max, int_min, bndry = 2 ** 31 - 1, -2 ** 31, 2 ** 31 // 10
        if not str: return 0  # 空字符串，提前返回
        while str[i] == ' ':
            i += 1
            if i == length: return 0  # 字符串全为空格，提前返回
        if str[i] == '-': sign = -1
        if str[i] in '+-': i += 1
        for c in str[i:]:
            if not '0' <= c <= '9': break
            if res > bndry or res == bndry and c > '7':
                return int_max if sign == 1 else int_min
            res = 10 * res + ord(c) - ord('0')
        return sign * res
# 20. Valid Parentheses.Given a string s containing just the characters '(', ')', '{', '}', '[' and ']', determine if the input string is valid.
# An input string is valid if:
# Open brackets must be closed by the same type of brackets.
# Open brackets must be closed in the correct order.
class Solution:
    def isValid(self, s: str) -> bool:
        stack = []
        if not s or len(s) % 2 == 1:
            return False
        for string in s:
            if string in ['(', '{', '[']:
                stack.append(string)
            else:
                if not stack:
                    return False
                if string == ')' and stack[-1] != '(' or string == '}' and stack[-1] != '{' or string == ']' and stack[
                    -1] != '[':
                    return False
                stack.pop()
        return not stack
#144. Binary Tree Preorder Traversal
# #Given the root of a binary tree, return the preorder traversal of its nodes' values.
class Solution:
    def preorderTraversal(self, root: TreeNode) -> List[int]:# root，left， right
        if not root:
            return []
        stack =[root]
        res = []
        while stack:
            node = stack.pop()
            res.append(node.val)
            if node.right:
                stack.append(node.right)
            if node.left:
                stack.append(node.left)
        return res
# 227. Basic Calculator II； Given a string s which represents an expression, evaluate this expression and return its value. 
# The integer division should truncate toward zero.
# You may assume that the given expression is always valid. All intermediate results will be in the range of [-231, 231 - 1].
# Note: You are not allowed to use any built-in function which evaluates strings as mathematical expressions, such as eval().
class Solution:
    def calculate(self, s: str) -> int:
        stack = []
        sign ='+'
        n = len(s)
        num = 0
        for i in range(n):
            if s[i]!=' ' and s[i].isdigit():
                num = 10*num+ord(s[i])-ord('0')
            if i ==n-1 or s[i] in '+-*/':
                if sign =='+':
                    stack.append(num)
                elif sign =='-':
                    stack.append(-num)
                elif sign =='*':
                    stack.append(stack.pop()*num)
                else:
                    stack.append(int(stack.pop()/num))
                sign =s[i]
                num = 0
        return sum(stack)
#3. Longest Substring Without Repeating Characters；
# Given a string s, find the length of the longest substring without repeating characters.
"""滑动窗口，时间复杂度O(n)"""
class Solutions:
    def lengthOfLongestSubstring(self, s: str) -> int:
        visited = set()
        left, right = 0,0
        max_len = 0
        n = len(s)
        for left in range(n):
            while right<n and s[right] not in visited:
                visited.add(s[right])
                right+=1
            max_len = max(max_len, right-left)
            visited.discard(s[left])
        return max_len
# 143. Reorder List。You are given the head of a singly linked-list. The list can be represented as:
# L0 → L1 → … → Ln - 1 → Ln
# Reorder the list to be on the following form:
# L0 → Ln → L1 → Ln - 1 → L2 → Ln - 2 → …
# You may not modify the values in the list's nodes. Only nodes themselves may be changed.
class ListNode:
    def __init__(self, val = 0, next = None):
        self.val = val
        self.next = next
class Solutions:
"""方法一：因为Linkedlist无法查询下标，采取使用list的方式，存储linkedlist，通过list来查询下标，重组新的linkedlist，
时间复杂度O(n),时间复杂度O(n)
"""
    def reorderList(self, head: ListNode) -> None:
        if not head:
            return None
        store_list = []
        node = head
        while node:
            node = node.next
            store_list.append(node)
        n = len(store_list)
        left, right = 0, n-1
        while left < right:
            store_list[left].next = store_list[right]
            left+=1
            if store_list[left] == store_list[right]:
                break
            store_list[right].next = store_list[left]
            right-=1
        store_list[left].next = None
"""方法二：可以把问题分解：1）先找到链表的中点；2）将链表的右半段反转；3）然后再合并链表两端"； 时间复杂度O(n),空间复杂度O(1)"""
    def reorderList(self, head: ListNode) -> None:
        if not head:
            return
        middle = self.find_middle(head)
        l1 = head
        l2 = middle.next
        middle.next = None
        l2 = self.reverse_list(l2)
        self.merge_two_list(l1,l2)
    def find_middle(self, head):
        p1, p2 = head, head
        while p2 and p2.next:
            p1 = p1.next
            p2 = p2.next.next
        return p1
    def reverse_list(self, head):
        pre = None
        cur = head
        while cur:
            tmp = cur.next
            cur.next = pre
            pre = cur
            cur = tmp
        return pre
    def merge_two_list(self, l1, l2):
        while l1 and l2:
            tmp1 = l1.next
            tmp2 = l2.next
            l1.next = l2
            l1 = tmp1
            l2.next = l1
            l2 = tmp2
# 118. Pascal's Triangle。 Given an integer numRows, return the first numRows of Pascal's triangle.
# In Pascal's triangle, each number is the sum of the two numbers directly above it as shown:
class Solution:
    def generate(self, numRows: int) -> List[List[int]]:
        tmp =[]
        dp = []
        for i in range(numRows):
            tmp = [1]*(i+1)
            j =1
            while j<i:
                tmp[j] = dp[i-1][j-1]+dp[i-1][j]
                j+=1
            dp.append(tmp)
        return dp
"""
MircoSoft coding 
"""
#4. Median of Two Sorted Arrays。Given two sorted arrays nums1 and nums2 of size m and n respectively, return the median of the two sorted arrays.
# The overall run time complexity should be O(log (m+n)).
class Solution:
    def findMedianSortedArrays(self, nums1: List[int], nums2: List[int]) -> float:
           """
            - 主要思路：要找到第 k (k>1) 小的元素，那么就取 pivot1 = nums1[k/2-1] 和 pivot2 = nums2[k/2-1] 进行比较
            - 这里的 "/" 表示整除
            - nums1 中小于等于 pivot1 的元素有 nums1[0 .. k/2-2] 共计 k/2-1 个
            - nums2 中小于等于 pivot2 的元素有 nums2[0 .. k/2-2] 共计 k/2-1 个
            - 取 pivot = min(pivot1, pivot2)，两个数组中小于等于 pivot 的元素共计不会超过 (k/2-1) + (k/2-1) <= k-2 个
            - 这样 pivot 本身最大也只能是第 k-1 小的元素
            - 如果 pivot = pivot1，那么 nums1[0 .. k/2-1] 都不可能是第 k 小的元素。把这些元素全部 "删除"，剩下的作为新的 nums1 数组
            - 如果 pivot = pivot2，那么 nums2[0 .. k/2-1] 都不可能是第 k 小的元素。把这些元素全部 "删除"，剩下的作为新的 nums2 数组
            - 由于我们 "删除" 了一些元素（这些元素都比第 k 小的元素要小），因此需要修改 k 的值，减去删除的数的个数
           """
        total_len = len(nums1) + len(nums2)
        if total_len % 2 == 1:
            k = (total_len + 1) // 2
            return self.find_kth_smallest(nums1, nums2, k)
        else:
            k1 = total_len // 2
            k2 = k1 + 1
            return (self.find_kth_smallest(nums1, nums2, k1) + self.find_kth_smallest(nums1, nums2, k2)) / 2
    def find_kth_smallest(self, nums1, nums2, k):
        m, n = len(nums1), len(nums2)
        idx1, idx2 = 0, 0
        while True:
            # 定义边界，也是递归出口
            # 如果idx1走出了nums1的边界了，还需要找到第k个最小的，就一定在第二个数组中。
            if m == idx1:
                return nums2[idx2 + k - 1]
            # 如果idx2走出了nums2的边界了，还需要找到第k个最小的，就一定在第一个数组中。
            if n == idx2:
                return nums1[idx1 + k - 1]
            # 最后定义k==1的情况，这三个出口之间前两个出口要早于第三个出口。
            if k == 1:
                return min(nums1[idx1], nums2[idx2])
            # 非边界情况
            idx_pivot1 = min(idx1 + k // 2 - 1, m - 1)  # 确保新的pivot1没走出来(包含边界)
            idx_pivot2 = min(idx2 + k // 2 - 1, n - 1)  # 确保新的pivot2没走出来(包含边界)
            pivot1, pivot2 = nums1[idx_pivot1], nums2[idx_pivot2]
            if pivot1 <= pivot2:  # 如果piovt1小于pivot2, 则可以抛弃比他小的k//2-1个数,通过迭代k的值从而来迭代pivot1的idx_pivot1进行前进
                k = k - (idx_pivot1 - idx1 + 1)
                idx1 = idx_pivot1 + 1  # 类似二分法，把起点向前挪
            else:
                k = k - (idx_pivot2 - idx2 + 1)
                idx2 = idx_pivot2 + 1
# 42. Trapping Rain Water
# Given n non-negative integers representing an elevation map where the width of each bar is 1, compute how much water it can trap after raining.
"""
方法一：动态规划
某个位置i，该地方的储水量等于min(max_left_height, min_right_height) - height[i]。动态规划的方法，分别左边和右边找出其昨天max_left_height和右边min_right_height，然后进行求和
时间复杂度是O(n)，空间复杂度O(n)
"""
class Solutions:
    def trap(self, height: List[int]) -> int:
        if not height:
            return 0
        # 分别构建左右两个高度list
        n = len(height)
        left_max = [0]*n
        right_max = [0]*n
        # 判断边界情况
        left_max[0] = height[0]
        right_max[n-1] = height[n-1]
        #正常情况
        for i in range(1,n):
            left_max[i] = max(left_max[i-1], height[i])
        for i in range(n-2,-1,-1):
            right_max[i] = max(right_max[i+1],height[i])
        #算出最终存水量
        res =[]
        for i in range(n):
            trap_water_each_point = min(left_max[i],right_max[i]) - height[i]
            res.append(trap_water_each_point)
        return sum(res)
"""
方法二：动态规划需要左右进行扫，并且同时还需要构建一个序列存储。通过双指针，不需要单独构建序列，时间复杂度依然是O(n),但是空间复杂度是O(1)
"""
   def trap(self, height: List[int]) -> int:
       if not height:
           return 0
       n = len(height)
       left, right = 0, n-1
       left_max, right_max =0, 0
       res = 0
       while left<right:
           left_max = max(height[left], left_max)
           right_max = max(height[right], right_max)
           if height[left]<height[right]:
               #如果左边高度比右边低，就计算左边的储数量，等左边最高位置减去该位置的高度，然后左边向前走一步
               res += left_max - height[left]
               left+=1
           else:
               res +=right_max - height[right]
               right-=1
        return res
# 23. Merge k Sorted Lists:You are given an array of k linked-lists lists, each linked-list is sorted in ascending order.
# Merge all the linked-lists into one sorted linked-list and return it.
# heapq: 最小顶堆结构，叶子节点大于根节点。 可以直接使用list进行初始化
# heapq.heappush(heap,item): 向heapq序列中插入元素，时间复杂度是O(logN)
# heapq.heappop(heap): 从heapq中弹出第一个元素（最小值），时间复杂度是O(logN)
# heapq.heapify(arr): 将序列转化成heapq序列，时间复杂度O(NlogN)
# heapq.nlargest(n, iterable, key = None)：从长序列中取出来最大的N个元素，时间复杂度O(NlogN)
# heapq.nsmallest(n, iterable, key = None)：从长序列中取出来最小的N个元素，时间复杂度O(NlogN)
class ListNode:
    def __init__(self, val = 0, next = None):
        self.val = val
        self.next = next
# 方法一：存入最小堆 ，然后弹出，时间复杂度O(KlogN)
class Solution:
    def mergeKLists(self, lists):
        import heapq
        if not lists:
            return None
        dummy = ListNode(0)
        cur = dummy
        head_heap = []
        for x in lists:
            while x:
                heapq.heappush(head_heap,x.val)
                x = x.next
        while head_heap:
            cur.next = ListNode(heapq.heappop(head_heap))
            cur  = cur.next
        return dummy.next
# 方法二： 两两合并对1进行哟花，时间复杂度O(NlogK);
# 时间复杂度分析: K条链表总节点数是N，平均每条链表是N/K个节点，因此合并两条链表的时间复杂度是O(N/K).
# 从K条开始两两合并成1条链表，因此每条链表会被合并logK次，因此K条链表会被合并K*logK次，总共时间复杂度是
# K*logK*N/K 即O(NlogK)
class ListNodes:
    def __init__(self, val=0, next = None):
        self.val = val
        self.next = next
class Solutions:
    def mergeKLists(self, lists):
        if not lists:
            return None
        n = len(lists)
        head = lists[0]
        for i in range(1, n):
            head = self.merge_two_lists(head, lists[i])
        return head
    def merge_two_lists(self, l1, l2):
        # 时间复杂度等于最短的链表的长度min_lenth, O(min_lenth).
        if not l1:
            return l2
        if not l2:
            return l1
        dummy = ListNode(0)
        tmp = dummy
        tmp1 = l1
        tmp2 = l2
        while tmp1 and tmp2:
            if tmp1.val < tmp2.val:
                tmp.next = tmp1
                tmp1 = tmp1.next
            else:
                tmp.next = tmp2
                tmp2 = tmp2.next
            tmp = tmp.next
        if tmp1:
            tmp.next = tmp1
        if tmp2:
            tmp.next = tmp2
        return dummy.next
# 297. Serialize and Deserialize Binary Tree；Serialization is the process of converting a data structure or object into a sequence of bits so that it can be stored in a file or memory buffer, or transmitted across a network connection link to be reconstructed later in the same or another computer environment.
# Design an algorithm to serialize and deserialize a binary tree. There is no restriction on how your serialization/deserialization algorithm should work. You just need to ensure that a binary tree can be serialized to a string and this string can be deserialized to the original tree structure.
# Clarification: The input/output format is the same as how LeetCode serializes a binary tree. You do not necessarily need to follow this format, so please be creative and come up with different approaches yourself.
class TreeNode:
    def __init__(self, x):
        self.val = x
        self.left = None
        self.right= None
# 方法一: DFS的范式，时间复杂度O(n), 空间复杂度O(n)
class Codec:
    def serialize(self, root):
        if not root:
            return 'None'
        return str(root.val)+','+str(self.serialize(root.left))+','+str(self.serialize(root.right))
    def deserialize(self, data):
        data_list = data.split(',')
        def dfs(data_list):
            eleme = data_list.pop(0)
            if eleme =='None':
                return None
            root = TreeNode(int(eleme))
            root.left = dfs(data_list)
            root.right = dfs(data_list)
            return root
        return dfs(data_list)
# 方法二: BFS的范式，时间复杂度O(n), 空间复杂度O(n)
class Codec:
    def serialize(self, root):
        if not root:
            return ''
        queue = deque([root])
        res = []
        while queue:
            node = queue.popleft()
            if node:
                res.append(str(node.val))
                queue.append(node.left)
                queue.append(node.right)
            else:
                res.append('None')
        return '[' + ','.join(res) + ']'
    def deserialize(self, data):
        if not data:
            return None
        data_list = data[1:-1].split(',')
        root = TreeNode(data_list[0])
        queue = deque([root])
        i = 1
        while queue:
            node = queue.popleft()
            if data_list[i] != 'None':
                node.left = TreeNode(data_list[i])
                queue.append(node.left)
            i += 1
            if data_list[i] != 'None':
                node.right = TreeNode(data_list[i])
                queue.append(node.right)
            i += 1
        return root
# 41. First Missing Positive。 Given an unsorted integer array nums, return the smallest missing positive integer.
# You must implement an algorithm that runs in O(n) time and uses constant extra space.
class Solution:
    def firstMissingPositive(self, nums: List[int]) -> int:
        n = len(nums)
        for i in range(n):
            if nums[i] <= 0:
                nums[i] = n + 1
        for i in range(n):
            num = abs(nums[i])
            if num <= n:
                nums[num - 1] = -abs(nums[num - 1])
        for i in range(n):
            if nums[i] > 0:
                return i + 1
        return n + 1
# 127. Word Ladder，A transformation sequence from word beginWord to word endWord using a dictionary wordList is a sequence of words beginWord -> s1 -> s2 -> ... -> sk such that:
# Every adjacent pair of words differs by a single letter.
# Every si for 1 <= i <= k is in wordList. Note that beginWord does not need to be in wordList.sk == endWord
# Given two words, beginWord and endWord, and a dictionary wordList, return the number of words in the shortest transformation sequence from beginWord to endWord, or 0 if no such sequence exists.
from collections import deque
class Solution:
    def ladderLength(self, beginWord: str, endWord: str, wordList: List[str]) -> int:
        if not wordList:
            return 0
        queue = deque([beginWord])
        wordList = set(wordList)
        visited = set([beginWord])
        distance = 0
        while queue:
            distance +=1
            for i in range(len(queue)):
                word = queue.popleft()
                if word == endWord:
                    return distance
                for next_word in self.nextWord(word):
                    if next_word not in wordList or next_word in visited:
                        continue
                    queue.append(next_word)
                    visited.add(next_word)
        return 0
    def nextWord(self, startWord):
        n = len(startWord)
        next_words = []
        for i in range(n):
            left, right = startWord[:i], startWord[i+1:]
            for char in 'abcdefghijklmnopqrstuvwxyz':
                if startWord[i] == char:
                    continue
                next_words.append(left+char+right)
        return next_words
#295. Find Median from Data Stream. The median is the middle value in an ordered integer list. If the size of the list is even,
# there is no middle value and the median is the mean of the two middle values.
# For example, for arr = [2,3,4], the median is 3.
# For example, for arr = [2,3], the median is (2 + 3) / 2 = 2.5.
# Implement the MedianFinder class:
# MedianFinder() initializes the MedianFinder object.
# void addNum(int num) adds the integer num from the data stream to the data structure.
# double findMedian() returns the median of all elements so far. Answers within 10-5 of the actual answer will be accepted.
# 方法一：通过双优先队列进行处理
class MedianFinder:
    def __init__(self):
        self.queMin = list()
        self.queMax = list()
    def addNum(self, num: int) -> None:
        queMin_ = self.queMin
        queMax_ = self.queMax
        if not queMin_ or num <= -queMin_[0]:
            heapq.heappush(queMin_, -num)
            if len(queMax_) + 1 < len(queMin_):
                heapq.heappush(queMax_, -heapq.heappop(queMin_))
        else:
            heapq.heappush(queMax_, num)
            if len(queMax_) > len(queMin_):
                heapq.heappush(queMin_, -heapq.heappop(queMax_))
    def findMedian(self) -> float:
        queMin_ = self.queMin
        queMax_ = self.queMax
        if len(queMin_) > len(queMax_):
            return -queMin_[0]
        return (-queMin_[0] + queMax_[0]) / 2
# 76. Minimum Window Substring
#Given two strings s and t of lengths m and n respectively, return the minimum window substring of s such that every character in t (including duplicates) is included in the window. If there is no such substring, return the empty string "".
#The testcases will be generated such that the answer is unique.
#A substring is a contiguous sequence of characters within the string.
class Solution:
    def minWindow(self, s: str, t: str) -> str:
        need_dict=collections.defaultdict(int)
        #use dictionary to store character in string t
        #and counter of the number of need
        for char in t:
            need_dict[char]+=1
        need_count = len(t)
        # first pointer
        i = 0
        res =(0,float('inf'))
        for j, char in enumerate(s): # j is the second pointer
            # only char in need_dict shall the need_count minus 1
            if need_dict[char]>0:
                need_count -=1
            need_dict[char] -=1
            # when need_count equals 0, all chars in t are included
            # so we need to remove the chars not in t
            if need_count==0:
                while True:
                    char = s[i]
                    # if char in need_dict and is 0, should break, there would be no extra words that can be removed
                    if need_dict[char] ==0:
                        break
                    need_dict[char]+=1
                    i+=1
                if j-i < res[1]-res[0]:
                    res = (i,j)
                need_dict[s[i]]+=1
                need_count+=1
                i+=1
        return '' if res[1]>len(s) else s[res[0]:res[1]+1]
# 44. Wildcard Matching
#Given an input string (s) and a pattern (p), implement wildcard pattern matching with support for '?' and '*' where:
#'?' Matches any single character.
#'*' Matches any sequence of characters (including the empty sequence).
#The matching should cover the entire input string (not partial).
class Solution:
    def isMatch(self, s: str, p: str) -> bool:
        m, n = len(s), len(p)
        dp = [[False] * (n + 1) for _ in range(m + 1)]
        dp[0][0] = True
        for i in range(1, n + 1):
            if p[i - 1] == '*':
                dp[0][i] = True
            else:
                break
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if p[j - 1] == '*':
                    dp[i][j] = dp[i][j - 1] or dp[i - 1][j]
                elif p[j - 1] == '?' or s[i - 1] == p[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1]
        return dp[m][n]
#224. Basic Calculator
# Given a string s representing a valid expression, implement a basic calculator to evaluate it, and return the result of the evaluation.
# Note: You are not allowed to use any built-in function which evaluates strings as mathematical expressions, such as eval().
# Constraints:
# 1 <= s.length <= 3 * 105
# s consists of digits, '+', '-', '(', ')', and ' '.
# s represents a valid expression.
# '+' is not used as a unary operation.
# '-' could be used as a unary operation and in this case, it will not be used directly after a +ve or -ve signs (will be inside parentheses).
# There will be no two consecutive operators in the input.
# Every number and running calculation will fit in a signed 32-bit integer.
class Solution:
    def calculate(self, s: str) -> int:
    #操作的步骤是：
    #如果当前是数字，那么更新计算当前数字；
    #如果当前是操作符+或者-，那么需要更新计算当前计算的结果 res，并把当前数字 num 设为 0，sign 设为正负，重新开始；
    #如果当前是 ( ，那么说明遇到了右边的表达式，而后面的小括号里的内容需要优先计算，所以要把 res，sign 进栈，更新 res 和 sign 为新的开始；
    #如果当前是 ) ，那么说明右边的表达式结束，即当前括号里的内容已经计算完毕，所以要把之前的结果出栈，然后计算整个式子的结果；
     #最后，当所有数字结束的时候，需要把最后的一个 num 也更新到 res 中。
        res, num, sign = 0, 0, 1
        stack = []
        for c in s:
            if c.isdigit():
                num = 10 * num + int(c)
            elif c == "+" or c == "-":
                res += sign * num
                num = 0
                sign = 1 if c == "+" else -1
            elif c == "(":
                stack.append(res)
                stack.append(sign)
                res = 0
                sign = 1
            elif c == ")":
                res += sign * num
                num = 0
                res *= stack.pop()
                res += stack.pop()
        res += sign * num
        return res
# 54. Spiral Matrix， Given an m x n matrix, return all elements of the matrix in spiral order.
class Solution:
    def spiralOrder(self, matrix: List[List[int]]) -> List[int]:
        if not matrix:
            return []
        m, n = len(matrix), len(matrix[0])
        # construct a list with right/down/left/up direction move which the direction need for spiral
        # for x: only left and right; for y: only up and down
        dx = [0, 1, 0, -1]
        dy = [1, 0, -1, 0]
        x = y = 0
        visited = set()
        di = 0  # initial direction di
        res = []
        for i in range(m * n):
            res.append(matrix[x][y])  # store visited point
            visited.add((x, y))  # store the coordinates into visited set
            tx, ty = x + dx[di], y + dy[di]  # caculate the next step coordinate
            if 0 <= tx < m and 0 <= ty < n and (tx, ty) not in visited:  # when no need to change the directions
                x, y = tx, ty
            else:  # when direction need to be changed
                di = (di + 1) % 4
                x, y = x + dx[di], y + dy[di]
        return res
# 200. Number of Islands
#Given an m x n 2D binary grid grid which represents a map of '1's (land) and '0's (water), return the number of islands.
#An island is surrounded by water and is formed by connecting adjacent lands horizontally or vertically. You may assume all four edges of the grid are all surrounded by water.
"""BFS和Graph
Two conditions that using BFS:
1) graph serach: from point to plane search, level search 
2) shortest path of graph 
"""
class Solution:
    def numIslands(self, grid: List[List[str]]) -> int:
        if not grid or not grid[0]:
            return 0
        islands =0
        m, n = len(grid), len(grid[0])
        for i in range(m):
            for j in range(n):
                if grid[i][j] =='0':
                    continue
                self.bfs(grid,i,j)
                islands+=1
        return islands
    def bfs(self, grid, x, y):
        m, n = len(grid), len(grid[0])
        queue = deque([(x,y)])
        while queue:
            x, y = queue.popleft()
            if 0<=x<m and 0<=y<n and grid[x][y] =='1': # when the coordinate is island
                grid[x][y] = '0' #turn the island to 0
                queue+=[[x+1, y],[x,y-1],[x-1,y],[x,y+1]] #search the other four directions
# 138. Copy List with Random Pointer
#A linked list of length n is given such that each node contains an additional random pointer, which could point to any node in the list, or null.
#Construct a deep copy of the list. The deep copy should consist of exactly n brand new nodes, where each new node has its value set to the value of its corresponding original node. Both the next and random pointer of the new nodes should point to new nodes in the copied list such that the pointers in the original list and copied list represent the same list state. None of the pointers in the new list should point to nodes in the original list.
#For example, if there are two nodes X and Y in the original list, where X.random --> Y, then for the corresponding two nodes x and y in the copied list, x.random --> y.
#Return the head of the copied linked list.
#The linked list is represented in the input/output as a list of n nodes. Each node is represented as a pair of [val, random_index] where:
#val: an integer representing Node.val
#random_index: the index of the node (range from 0 to n-1) that the random pointer points to, or null if it does not point to any node.
#Your code will only be given the head of the original linked list.
"""
# Definition for a Node.
class Node:
    def __init__(self, x: int, next: 'Node' = None, random: 'Node' = None):
        self.val = int(x)
        self.next = next
        self.random = random
"""
class Solution:
    def copyRandomList(self, head: 'Node') -> 'Node':
        if not head:
            return head
            # copy all nodes to insert after the original node
        cur = head
        while cur:
            cur.next = Node(cur.val, cur.next, None)
            cur = cur.next.next  # move two steps to insert after each original node
        # connect all random nodes of each copied node
        cur = head
        copy_head = head.next
        while cur:
            if cur.random:
                cur.next.random = cur.random.next
            cur = cur.next.next
            # diconnect the connection between original linked-list and copy linked-list
        cur = head
        cur_ = copy_head
        while cur and cur_:
            cur.next = cur_.next
            cur = cur.next
            if cur:
                cur_.next = cur.next
            cur_ = cur_.next
        return copy_head
# 1448. Count Good Nodes in Binary Tree,root to X there are no nodes with a value greater than X.
# Return the number of good nodes in the binary tree.
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def goodNodes(self, root: TreeNode) -> int:
        self.counter = 0
        max_value = -sys.maxsize
        self.dfs(root, max_value)
        return self.counter
    def dfs(self, node, max_value):
        if not node:
            return
        if node.val >= max_value:
            self.counter += 1
            max_value = node.val
        self.dfs(node.left, max_value)
        self.dfs(node.right, max_value)
        return
# 151. Reverse Words in a String. Given an input string s, reverse the order of the words.
# A word is defined as a sequence of non-space characters. The words in s will be separated by at least one space.
# Return a string of the words in reverse order concatenated by a single space.
# Note that s may contain leading or trailing spaces or multiple spaces between two words. The returned string should only have a single space separating the words. Do not include any extra spaces.
"""two pointers: time complexity is O(N), space complexity is O(N)"""
class Solution:
    def reverseWords(self, s: str) -> str:
        s = s.strip() # remove the empty at the begin and end of string
        i = j = len(s) -1 # two pointers from the end of the string
        res =[]
        while i>=0:
            while i>=0 and s[i]!=' ': # move i from the end to the start of the last word in s
                i-=1
            res.append(s[i+1:j+1]) # append the last word
            while s[i] ==' ': # move i through the empty between words
                i-=1
            j=i # keep the i and j at the end of each word
        return ' '.join(res)
# 48. Rotate Image. You are given an n x n 2D matrix representing an image, rotate the image by 90 degrees (clockwise).
# You have to rotate the image in-place, which means you have to modify the input 2D matrix directly. DO NOT allocate another 2D matrix and do the rotation.
class Solution:
    def rotate(self, matrix: List[List[int]]) -> None:
        """
        Do not return anything, modify matrix in-place instead.
        """
        n = len(matrix)
        # rotate on the horizon, which means exchange the upper half row  with the down half rows
        for i in range(n//2):
            for j in range(n):
                matrix[i][j], matrix[n-i-1][j] = matrix[n-i-1][j], matrix[i][j]
        # rotate the diagonal, which means exchange the upper side of diagonal with the down side of diagonal
        for i in range(n):
            for j in range(i):
                matrix[i][j], matrix[j][i] = matrix[j][i], matrix[i][j]
# 33. Search in Rotated Sorted Array.There is an integer array nums sorted in ascending order (with distinct values).
# Prior to being passed to your function, nums is possibly rotated at an unknown pivot index k (1 <= k < nums.length) such that the resulting array is [nums[k], nums[k+1], ..., nums[n-1], nums[0], nums[1], ..., nums[k-1]] (0-indexed). For example, [0,1,2,4,5,6,7] might be rotated at pivot index 3 and become [4,5,6,7,0,1,2].
# Given the array nums after the possible rotation and an integer target, return the index of target if it is in nums, or -1 if it is not in nums.
# You must write an algorithm with O(log n) runtime complexity.
"""binary search
two conditions: 1) the list is ascending between[start, mid]; 
2)the list is acending between [mid+1, end]
Time complexity is O(N), Space complexity is O(1)
"""
class Solution:
    def search(self, nums: List[int], target: int) -> int:
        if not nums:
            return -1
        start, end = 0, len(nums)-1
        while start<=end:
            mid = (start+end)//2
            if nums[mid] == target:
                return mid
            if nums[mid] >= nums[start]:
                if nums[start]<=target<nums[mid]:
                    end = mid-1
                else:
                    start = mid+1
            else:
                if nums[mid]<target<=nums[end]:
                    start = mid+1
                else:
                    end = mid-1
        return -1
# 56. Merge Intervals.Given an array of intervals where intervals[i] = [starti, endi], merge all overlapping intervals, and return an array of the non-overlapping intervals that cover all the intervals in the input.
# Given an array of intervals where intervals[i] = [starti, endi], merge all overlapping intervals, and return an array of the non-overlapping intervals that cover all the intervals in the input.
"""
Time complexity O(nlogn) the cost of time mainly from the sort of array;
Space complexity O(logn)
"""
class Solution:
    def merge(self, intervals: List[List[int]]) -> List[List[int]]:
        intervals.sort(key = lambda x:x[0]) # sort the left boundary of each interval
        res = []
        for interval in intervals:
            if not res or res[-1][1]<interval[0]: # if the left boundary of last interval in res less than current interval or res is empty, just add the interval into the res
                res.append(interval)
            else: # if the left boundary of last interval in res equal to the left of interval, merge them with the max right boundary between them
                res[-1][1] = max(res[-1][1],interval[1])
        return res
"""
"""
# 1) 最短路径，从grid(m*n)的左上角一直到右下角，self.top =1, 上面就有连接，0就没有连接； self.left =1 左边有链接，0 没有链接
# 2）二叉树对称 ； 3）足球比赛，多少种可能；4）中文字符转化为int