import random,numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from skimage.measure import compare_ssim as ssim
import cv2
import random
import math
from numba import jit

def black(image, no_of_points):

	'''
	Dimensions no_of_points X 2

	'''
    #black = []

    '''
    rand_rows = random sample n_points from (n_rows) [0,3,4...]
	rand_cols = random sample n_points from (n_cols)
	
	image[rand_rows, rand_cols] = 0
	return image
    '''
    for points in range(no_of_points):
        rand_row = random.randint(0,len(image)- 1)
        rand_col = random.randint(0,len(image[0]) - 1)
        if((image[rand_row][rand_col] == 1)):
            image[rand_row][rand_col] = 0
            #black.append((rand_row,rand_col))
        else:
            points -= 1
    return image


def createImage(row, col, no_of_points):
    
    # Z = np.random.random((row,col)) # mention the range of values you want to have them in
    img = np.ones((row,col))
    
    #blac = black(img, no_of_points)
    img = black(img, no_of_points)
    """
    for cord in blac:
        img[cord[0], cord[1]] = [0,0,0]
    """
    return img


def newPopulation(pop_max, target_img, num_of_points, target):
    
    row = len(target)
    col = len(target[0])
    population = []
    # can create individual processes if lags
    for image in range(pop_max):
        im = createImage(row, col, num_of_points)
        population.append(im)
        
    return population


def fitness(image, target):
    
    # we run this on GPU, if need be
    """
    ms = mse(image, target)
    score = 1/(10**(ms))
    return (score**4)*100
    
    """
    score = 0
    for row in range(len(image)):
        for col in range(len(image[0])):
            if(((image[row][col] == 0) and (target[row][col] == 0)) ):
                score += 2
            

    return (score)
    # Experimental fitness below

    
def populationFitness(population, target):
    """
    score = []
    min_score = 999
    for image in population:
        fitness_score = fitness(image, target)
        score.append(fitness_score)
        """"""
        if(fitness_score<min_score):
            min_score = fitness_score
        """"""
    
    for i in range(len(score)):
        #score[i] = score[i] - min_score + 1
        score[i] = int (score[i] )
    
    return [population, score]
    """
    score = []
    min_score = 9999999 # np.inf , float('inf')
    for image in population:
        fitness_score = math.exp(fitness(image, target))**2
        score.append(fitness_score)
        if(fitness_score<min_score):
            min_score = (fitness_score)
    
    for i in range(len(score)):
        score[i] = score[i] - min_score + 1
        score[i] = int (score[i] )
        
    return [population, score]
    
def matingPool(population_fitness):
    """
    population = population_fitness[0]
    score = population_fitness[1]
    pool = []
    for i in range(len(population)):
        for j in range(score[i]):
            pool.append(population[i])
    return pool
    """
    population = population_fitness[0]
    score = population_fitness[1]
    pool_norm_score = []
    maxm = 0
    for i in range(len(population)):
        if(score[i]>maxm):
            maxm = score[i]
    for i in range(len(population)):
        pool_norm_score.append(score[i]/maxm)
    
    return pool_norm_score
        
def createChild(img1, img2, num_of_points):
	# rewrite this code, please name it as crossover
    row = len(img1)
    col = len(img1[0])
    black = []
    Z = np.random.random((row,col))
    G = np.ones((row,col))
    num_points = 0
    for i in range(row):
        for j in range(col):
            if((img1[i][j] == 0) and (img2[i][j] == 0)):
                p = 0.95
            elif((img1[i][j] == 0) or (img2[i][j] == 0)):
                p = 0.4
            else:
                p = 0.05
            n = random.random()
            if(n<p):
                G[i][j] = 0
                black.append([i,j])
                num_points += 1
            if(num_points > num_of_points):
                break
    return G,black

import random
def createChildren(pool_norm_score, pop_max, num_of_points,old_population, target):
    
    population = []
    population_points =[]
    for i in range(pop_max):
        
        index1 = 0
        r = random.random()
        while(r>0):
            r = r - pool_norm_score[index1]
            index1+=1
        
        index1-=1
        
        index2 = 0
        r = random.random()
        while(r>0):
            r = r - pool_norm_score[index2]
            index2+=1
        
        index2-=1
        
       
        child,black = createChild(old_population[index1], old_population[index2], num_of_points)
        population.append(child)
        population_points.append(black)
    
    return population,population_points

def mutateImage(img,black,black_target):
    for i in range(2):
        while(True):
            rand_index = random.randint(0,len(black)-1)
            i_b, j_b = black[rand_index]
            if([i_b,j_b] not in black_target):
                img[i_b][j_b] = 1
                break
        while(True):
            row_rand = random.randint(0,len(img)-1)
            col_rand = random.randint(0,len(img[0])-1)

            if((img[row_rand][col_rand] == 1)):
                img[row_rand][col_rand] = 0
                break
            
        
        
        
    return img

def mutatePopulation(population, chance_of_mutation,population_points,black_target):
    for i in range(len(population)):
        rand = random.randint(0,100)
        black = population_points[i]
        if(rand < chance_of_mutation):
            population[i] = mutateImage(population[i],black,black_target)
    return population

import time
@jit
def main():
	start = time.time()

	#img = Image.open('test.png')
	img = cv2.imread('test.png',0)

	########## remove this code ###########
	black_target = []
	for i in range(len(img)):
	    for j in range(len(img[0])):
	        
	        
	        if(img[i][j] >20):

	            img[i][j] = 1
	        else:
	                img[i][j] = 0
	                black_target.append([i,j])
	#######################################

	#plt.imshow(newPopulation(3,target)[0], interpolation = "nearest")
	#plt.show()
	target = img
	pop_max = 50
	num_of_points = 10
	pop = newPopulation(pop_max, target, num_of_points,target)

	count=0
	flag = 1
	while(flag==1):
	    count+=1
	    
	    population_fitness = populationFitness(pop, target)
	    start = time.time()

            #img = Image.open('test.png')
	    pool = matingPool(population_fitness)

	    
	    pop,population_points=createChildren(pool, pop_max, num_of_points,population_fitness[0], target)
	    
	    pop=mutatePopulation(pop, 100, population_points,black_target)
	    print("no of generation:",count)
	    max = 0
	    max_i = pop[0]
	    for i in pop:
	        if(fitness(i, target) > 20):
	            flag = 0
	            print(count)
	            end = time.time()
	            print(end-start)
	        if(fitness(i, target) > max):
	            max = fitness(i, target)
	            max_i = i
	    print(fitness(max_i, target))
	    #plt.imshow(max_i, cmap = plt.cm.gray)
	    #plt.axis("off")
	    plt.imshow(max_i,interpolation='nearest')
	    plt.pause(0.00001)        
	plt.show()

main()
