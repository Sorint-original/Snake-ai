import numpy as np

class my_class(object):
    data_pointer = 0

    def __init__ (self,capacity) :
       
        self.capacity = capacity
        self.tree = np.zeros(2*capacity -1)
        self.data = np.zeros(capacity,dtype=object)

    def add(self,priority,data) :
        tree_index = self.data_pointer + self.capacity - 1
        
        # Update data frame
        self.data[self.data_pointer] = data

        # Update the leaf
        self.update (tree_index, priority)

        # Add 1 to data_pointer
        self.data_pointer += 1
        if self.data_pointer >= self.capacity:  
            self.data_pointer = 0      

    def update(self, tree_index, priority):
        # Change = new priority score - former priority score
        change = priority - self.tree[tree_index]
        self.tree[tree_index] = priority

        # then propagate the change through tree
        # this method is faster than the recursive loop
        while tree_index != 0:
            tree_index = (tree_index - 1) // 2
            self.tree[tree_index] += change






