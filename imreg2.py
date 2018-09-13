import random,numpy as np
from PIL import Image
import matplotlib.pyplot as plt
def black(image, no_of_points):

    black = []
    for points in range(no_of_points):
        rand_row = random.randint(0,len(image)- 1)
        rand_col = random.randint(0,len(image[0]) - 1)
        if((image[rand_row][rand_col] == [1,1,1]).all()):
            image[rand_row][rand_col] = [0,0,0]
            black.append((rand_row,rand_col))
        else:
            points -= 1
    return black


def createImage(row, col, no_of_points):
    
    Z = np.random.random((row,col))
    img = np.ones((row,col,3))
    
    blac = black(img, no_of_points)
    
    for cord in blac:
        img[cord[0], cord[1]] = [0,0,0]

    return img

def newPopulation(pop_max, target_img, num_of_points):
    
    row = len(target)
    col = len(target[0])
    population = []
    for image in range(pop_max):
        im = createImage(row, col, num_of_points)
        population.append(im)
        
    return population


def fitness(image, target):
    
    
    
    score = 0
    for row in range(len(image)):
        for col in range(len(image[0])):
            if(((image[row][col] == [0,0,0]).all() and (target[row][col] == [0,0,0]).all()) ):
                score += 1
            
    return score
    
def populationFitness(population, target):
    
    score = []
    for image in population:
        fitness_score = fitness(image, target)
        score.append(fitness_score)
    min_score = min(score)
    
    for i in range(len(score)):
        score[i] = score[i] - min_score + 1
        score[i] = int (score[i]**1.5 )
    return [population, score]

def matingPool(population_fitness):
    population = population_fitness[0]
    score = population_fitness[1]
    pool = []
    for i in range(len(population)):
        for j in range(score[i]):
            pool.append(population[i])
    return pool

import random 
def createChild(img1, img2, num_of_points):
    row = len(img1)
    col = len(img1[0])
    Z = np.random.random((row,col))
    G = np.ones((row,col,3))
    num_points = 0
    for i in range(row):
        for j in range(col):
            if((img1[i][j] == [0,0,0]).all() and (img2[i][j] == [0,0,0]).all()):
                p = 0.8
            elif((img1[i][j] == [0,0,0]).all() or (img2[i][j] == [0,0,0]).all()):
                p = 0.5
            else:
                p = 0.1
            n = random.random()
            if(n<p):
                img1[i][j] = [0,0,0]
                num_points += 1
            if(num_points > num_of_points):
                break

    
    # black_from_im1 = []
    # black_from_im2 = []
    # black1 = 0
    # while(black1 <= num_of_points / 2):
    #     rand_row = random.randint(0,len(img1)- 1)
    #     rand_col = random.randint(0,len(img1[0]) - 1)
    #     if((img1[rand_row][rand_col] == [0,0,0]).all()):
            
    #         black_from_im1.append((rand_row,rand_col))
    #         black1 += 1
    
    # black2 = 0
    # while(black2 <= num_of_points / 2):
    #     rand_row = random.randint(0,len(img2)- 1)
    #     rand_col = random.randint(0,len(img2[0]) - 1)
    #     if((img2[rand_row][rand_col] == [0,0,0]).all()):
            
    #         black_from_im2.append((rand_row,rand_col))
    #         black2 += 1
    # for cord in black_from_im1:
    #     G[cord[0], cord[1]] = [0,0,0]
    # for cord in black_from_im2:
    #     G[cord[0], cord[1]] = [0,0,0]
    return G


def createChildren(pool, pop_max, num_of_points):
    
    population = []
    for i in range(pop_max):
        
        rand1 = random.randint(0,len(pool)-1)
        rand2 = random.randint(0,len(pool)-1)
        child = createChild(pool[rand1], pool[rand2], num_of_points)
        population.append(child)
    
    return population



import random
def mutateImage(img):
    for i in range(100):
        row_rand = random.randint(0,len(img)-1)
        col_rand = random.randint(0,len(img[0])-1)
        
        if((img[row_rand][col_rand] == [0,0,0]).all()):
            img[row_rand][col_rand] = [1,1,1]
        else:
            img[row_rand][col_rand] = [0,0,0]
        
    return img

def mutatePopulation(population, chance_of_mutation):
    for i in range(len(population)):
        rand = random.randint(0,100)
        if(rand < chance_of_mutation):
            population[i] = mutateImage(population[i])
    return population

#main()
img = Image.open('test2.png')
target = np.array(img)
for i in range(len(target)):
    for j in range(len(target[0])):
        for k in range(len(target[0][0])):
                if(target[i][j][k] == 255):
                    
                    target[i][j][k] = 1
#plt.imshow(newPopulation(3,target)[0], interpolation = "nearest")
#plt.show()

pop_max = 100
num_of_points = 10
pop = newPopulation(pop_max, target, num_of_points)

count=1
flag = 1
while(flag == 1):
    count+=1
    population_fitness = populationFitness(pop, target)
    pool = matingPool(population_fitness)
    pop=createChildren(pool, pop_max, num_of_points)
    pop=mutatePopulation(pop, 1)
    print(count)
    max = 0
    max_i = pop[0]
    for i in pop:
        
        if(fitness(i, target) > 500 ):
            flag = 0
            print("no of generation:",count)
            break
        if(fitness(i, target) > max):
            max = fitness(i, target)
            max_i = i
    print(fitness(max_i, target))
    plt.imshow(max_i,interpolation='nearest')
    plt.pause(0.001)        
plt.show()

            
            
            
    
    
