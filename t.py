
# import numpy as np
# import matplotlib.pyplot as plt
# import random
# from sklearn.linear_model import LinearRegression


# # from tqdm import *
# # from time import sleep
 
 
# # bar = tqdm(range(0,100), colour = "green", ncols=100, desc="Process: ", unit="Byte")
# # for i in bar:
# #     sleep(.01)
# # print("Completed!")


# # x_val = []
# # y_val = []

# # reg = LinearRegression()

# # for i in range(1000):
# #     plt.clf()
# #     x_val.append(random.randint(0,100))
# #     y_val.append(random.randint(0,100))

# #     x = np.array(x_val)
# #     x = x.reshape(-1,1)

# #     y = np.array(y_val)
# #     y = y.reshape(-1,1)
        
# #     if i%10==0:
# #         reg.fit(x,y)
# #         plt.xlim(0,100)
# #         plt.ylim(0,100)
# #         plt.scatter(x_val,y_val,color="red")
# #         plt.plot((list(range(100))),reg.predict(np.array([x for x in range(100)]).reshape(-1,1)))
# #         plt.pause(0.001)

# # plt.show()

# def lengthOfLastWord(A):
#     A = A.rstrip()
#     n = len(A)
#     j=n-1
#     while j>-1:
#         if A[j]==" ":
#             break
#         j-=1
#     if j==n-1:
#         return 0
#     return (n-1-j)

# print(lengthOfLastWord("   Hello World  "))


def fn(a,b,n):
    r = round(b/a,3)
    # first = round(a/r)
    ans = b*(r**(n-3))
    return round(ans,3)

print(fn(1,2,4))