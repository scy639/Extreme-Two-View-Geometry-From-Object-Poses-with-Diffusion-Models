import os
def get_path(i,j,x,y,z):
    return f"{i}-{j}(x={x},y={y},z={z}).jpg"
def i2glob(i,):
    return f"{i}-*.jpg"


#------zero123 output_im------------------------
def parse_path(path:str):
    path=os.path.basename(path)
    """
      path=11-2(x=0,y=30.0,z=0).png=i-j(x=0,y=30.0,z=0).png
    """
    file=path
    i = int(file.split('-')[0])
    j = int(file.split('-')[1].split('(')[0])  # index of sample
    rest = file[len(f"{i}-j"):]
    x=float(rest.split('(')[1].split(',')[0].split('=')[1])
    y=float(rest.split('(')[1].split(',')[1].split('=')[1])
    z=float(rest.split('(')[1].split(',')[2].split('=')[1].split(')')[0])
    return i,j,x,y,z
def sort_outputIm_A(paths:list[str])->list[str]:
    def path2xy(imgPath):
        i,j,x,y,z=parse_path(imgPath)
        assert j==0
        return x,y
    l_xy=list(map(path2xy,paths))
    new_imgPaths=[]
    
    N=3
    min_x,max_x=min(l_xy,key=lambda xy:xy[0])[0],max(l_xy,key=lambda xy:xy[0])[0]
    l__path_xy=zip(paths,l_xy)
    #firstly we sort by y
    l__path_xy=sorted(l__path_xy,key=lambda path_xy:path_xy[1][1],reverse=False)
    #crate a emtpy list with len N
    l=[[] for i in range(N)]
    
    x_interval=(max_x-min_x)/N
    ranges=[[min_x+i*x_interval,min_x+i*x_interval+x_interval]      for i in range(N)] 
    ranges[-1][1]=max_x
    for path,xy in l__path_xy:
        x,y=xy
        level=int((x-min_x)//x_interval)
        if level==N:
            level-=1
        assert ranges[level][0]<=x<=ranges[level][1]
        l[level].append(path)
    l.reverse()
    # print(f"{l=}")
    #flat l
    new_imgPaths=[]
    for i in range(N):
        new_imgPaths.extend(l[i])
    return new_imgPaths