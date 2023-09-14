import pygame

class map_class(object):
    def __init__ (self,size) :
        self.size = size
        self.tile_map =[]
        for i in range(size):
            aux = []
            for j in range(size):
                aux.append((0,0))
            self.tile_map.append(aux)



