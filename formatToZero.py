import sys


def readIn():
    col1 = []
    col2 = []
    with open(sys.argv[1]) as a:
        d = a.next()
        for line in a:
            b = line.split()
            col1.append(int(b[0])-1)
            col2.append(int(b[1])-1)
    return col1, col2, d

def writeTo(col1, col2, d):
    with open(sys.argv[2], "w+") as b:
        b.write(d)
        for i in range(len(col1)):
            b.write(str(col1[i]) + " " + str(col2[i]) + "\n") 
    


def main():
    col1, col2, d = readIn()
    writeTo(col1, col2,d)




if __name__ == '__main__':
    main()
