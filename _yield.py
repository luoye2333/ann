def test():
    for i in range(5):
        yield i
        print("dads")
        yield i+1

x=test()
for i,c in enumerate(x):
    print(i,c)
    if(i==2):
        break

def genter():
    a = 4
    b = 5
    c = 6
    for i in range(5):
        yield a
        print('hhh'+str(i))
        yield b
        print("aaa" + str(i))
        yield c