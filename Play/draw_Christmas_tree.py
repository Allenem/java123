from turtle import *
import random


def draw_5pointedstar(lens):
    color('orange', 'yellow')
    begin_fill()
    left(126)          # turn left by 72+108/2 degree from upward
    for i in range(5):
        forward(lens)
        right(144)     # turn right by 180-36 degree
        forward(lens)
        left(72)
    end_fill()
    right(126)         # turn right by 72+108/2 degree to upward


def draw_light():
    if random.randint(0, 30) == 0:
        color('tomato')
        circle(6)
        color('dark green')
    elif random.randint(0, 30) == 1:
        color('orange')
        circle(3)
        color('dark green')
    else:
        color('dark green')


def draw_tree(d, s):
    if d <= 0: return
    forward(s)
    draw_tree(d-1, s*.8)
    right(120)
    draw_tree(d-3, s*.5)
    draw_light()
    right(120)
    draw_tree(d-3, s*.5)
    right(120)
    backward(s)


def draw_somecircles(num):
    for i in range(num):
        a = 200 - 400 * random.random()
        b = 10 - 20 * random.random()
        up()
        forward(b)
        left(90)
        forward(a)
        down()
        if random.randint(0, 1) == 0:
            color('tomato')
        else:
            color('wheat')
        circle(2)
        up()
        backward(a)
        right(90)
        backward(b)


def draw_snowflake(num):
    hideturtle()
    pensize(2)
    for i in range(num):
        pencolor('white')
        penup()
        setx(random.randint(-350,350))
        sety(random.randint(-100,350))
        pendown()
        dens = 6
        snowsize = random.randint(1,10)
        for j in range(dens): 
            forward(int(snowsize))
            backward(int(snowsize))
            right(int(360/dens)) 


if __name__ == '__main__':

    n = 100.0
    speed('fastest')
    screensize(bg='black')       # bacground color
    left(90)                     # turn left 90 degree from middle of img, from rightward, to the beginning of 5points star
    forward(3*n)
    # Step1 draw a 5pointstar at the top of the tree
    draw_5pointedstar(lens=n/5)
    # Step2 draw the tree with tomato or wheat circles
    color('dark green')
    backward(n*4.8)
    draw_tree(15, n)
    backward(n/2)
    # Step3 draw some circles at the bottom of tree as decoration
    draw_somecircles(num=50)
    # Step4 write some words
    color('dark red','red')
    write('Merry Christmas',align ='center',font=('Comic Sans MS',40,'bold'))
    # Step5 draw some snowflakes    
    draw_snowflake(num=50)

    done()