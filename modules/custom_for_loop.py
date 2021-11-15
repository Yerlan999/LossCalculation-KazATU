from time import sleep


k = 0
n = 0
il = ["a", "b", "c", "d"]


next_outer = False

update_inner = True

inner_loop = 50 # n
outer_loop = 560 # k
PR = 0

counter = 1

while True:

    PR += 1

    if k == outer_loop:
        break

    # print("Outer,  Inner, PR: ", n, k, PR, update_inner)

    # [block-1]
    print(counter); counter+=1


    if (k > 0) or (k == 0 and PR ==2):
        x = 0

        # [block-2]

    # [block-3]

    if update_inner:

        if n < inner_loop-1:
            n += 1
            next_outer = False
        else:
            n = 0
            next_outer = True

    if next_outer:
        if k < outer_loop:
            k += 1
        else:
            k = 0

    if k == 0 and PR == 1:
        update_inner = False  # ---> 1700
        continue
    if PR == 2:
        update_inner = True  # ---> 1500
        PR = 0
        continue


    # sleep(1)
