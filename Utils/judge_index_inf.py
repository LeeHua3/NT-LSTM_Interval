def ft3(a):
    if a == -1:
        return 1
    elif a < 2.625:
        return 0
    elif a > 5.705:
        return 2
    else:
        return 1


def ft4(a):
    if a == -1:
        return 1
    elif a < 9.005:
        return 0
    elif a > 19.045:
        return 2
    else:
        return 1


def tsh(a):
    if a == -1:
        return 1
    elif a < 0.3515:
        return 0
    elif a > 4.94:
        return 2
    else:
        return 1


def trab(a):
    if a == -1:
        return 1
    elif a < 1.755:
        return 0
    else:
        return 1

#缺失值-1,都默认返回为1（正常），因为正常的概率最大
# ft3_state=[2.625,5.705]
# ft4_state=[9.005,19.045]
# tsh_state=[0.3515,4.94]

index_state=[[2.625,5.705],
            [9.005,19.045],
            [0.3515,4.94]]

index_function=[ft3,ft4,tsh]

if __name__ == "__main__":
    state1=index_function[0](5.706)
    state2 = index_function[0](5.704)
    state3 = index_function[0](1)
    print(state1,state2,state3)