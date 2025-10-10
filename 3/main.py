from transform import *

"""
========================
A.2 Recover Homographies
========================
"""
im1 = '3/pair_pics/bancroft2.jpg'
im2 = '3/pair_pics/bancroft5.jpg'

# # im3 = '3/pair_pics/wheeler_2.jpg'
# # im4 = '3/pair_pics/wheeler_6.jpg'

# bancroft
pts_1 = np.array([(348,160),(419,137),(392,244),(478,244),(404,187),(544,231),(399,113),(361,254)])
pts_2 = np.array([(20,126),(146,127),(96,259),(209,248),(120,181),(268,233),(122,91),(29,284)])

# # wheeler 
# # {"im1_name":"wheeler_2","im2_name":"wheeler_6","im1Points":[[604,285],[515,280],[594,175],[614,13],[462,57],[648,121],[621,55]],"im2Points":[[181,280],[88,280],[179,173],[206,23],[41,24],[232,130],[212,66]]}
# pts_3 = np.array([(604,285),(515,280),(594,175),(614,13),(462,57),(648,121),(621,55)])
# pts_4 = np.array([(181,280),(88,280),(179,173),(206,23),(41,24),(232,130),(212,66)]) 

# wheeler_wide
# im1 = '3/pair_pics/wheeler_wide_3.jpg'
# im2 = '3/pair_pics/wheeler_wide_6.jpg'

# # wheeler wide
# # {"im1_name":"wheeler_wide_3","im2_name":"wheeler_wide_6","im1Points":[[352,64],[356,90],[389,65],[428,59],[349,183],[442,133],[484,132],[529,172]],"im2Points":[[111,18],[113,53],[166,36],[216,47],[95,172],[227,128],[268,133],[304,173]]}
# pts_1 = np.array([(352,64),(356,90),(389,65),(428,59),(349,183),(442,133),(484,132),(529,172)])
# pts_2 = np.array([(111,18),(113,53),(166,36),(216,47),(95,172),(227,128),(268,133),(304,173)]) 

# maclaughlin hall
# im1 = '3/pair_pics/mc2.jpg'
# im2 = '3/pair_pics/mc4.jpg'

# # maclaughlin hall
# # {"im1_name":"mc2","im2_name":"mc4","im1Points":[[231,386],[130,626],[230,659],[293,605],[318,543],[102,375],[230,417],[266,460]],"im2Points":[[237,44],[137,293],[225,320],[284,275],[315,226],[82,28],[232,87],[270,138]]}
# pts_1 = np.array([(231,386),(130,626),(230,659),(293,605),(318,543),(102,375),(230,417),(266,460)])
# pts_2 = np.array([(237,44),(137,293),(225,320),(284,275),(315,226),(82,28),(232,87),(270,138)])

# # home
# im1 = '3/pair_pics/home1.jpg'
# im2 = '3/pair_pics/home2.jpg'
# # home
# # {"im1_name":"home1","im2_name":"home2","im1Points":[[525,97],[505,155],[558,140],[560,40],[662,25],[668,137],[372,58],[494,13]],"im2Points":[[219,101],[200,158],[248,148],[253,51],[335,54],[336,148],[42,33],[191,13]]}
# pts_1 = np.array([(525,97),(505,155),(558,140),(560,40),(662,25),(668,137),(372,58),(494,13)])
# pts_2 = np.array([(219,101),(200,158),(248,148),(253,51),(335,54),(336,148),(42,33),(191,13)])

def homography():
    H = compute(pts_1, pts_2)
    # print(H)
    # display_points(im1, pts_1)
    # display_points(im2, pts_2)

    imwarped_nn = warpImageNearestNeighbor(im1, H)
    # imwarped_bil = warpImageBilinear(im2, H)
    # plt.imshow(imwarped_bil)
    # plt.axis('off')
    # plt.show()

    image = blend(im2, im1, H)
    # # save_image("mc", image)
    plt.imshow(image)
    plt.axis('off')
    plt.show()
    
homography()

# rectification
# card: [[169,287],[215,299],[271,254],[226,243]]
# nilla: [[83,291],[182,359],[265,324],[161,266]] 

card = '3/rectification/card.jpg'

card_pts = np.array([(169,287),(215,299),(271,254),(226,243)])
nilla_pts = np.array([(83,291),(182,359),(265,324),(161,266)])

card_goal = np.array([(170,280),(170,310),(220,310),(220,280)])
# nilla_pts = np.array()

jimin = '3/rectification/jimin.jpg'
points = np.array([(84,345),(144,414),(319,286),(233,244)])
goal = np.array([[96,428],[216,428],[216,234],[96,234]])

def rectify():
    H = compute(goal, points)
    display_points(jimin, points)

    imwarped_nn = warpImageNearestNeighbor(jimin, H)
    save_image("jimin_rectified", imwarped_nn)
    plt.imshow(imwarped_nn)
    plt.axis('off')
    plt.show()

# rectify()



