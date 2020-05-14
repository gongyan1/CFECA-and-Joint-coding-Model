import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator

index = [i for i in range(0,101)]
path = 'train_lane_acc_v1_2020-04-21-12:19:43.txt'
with open(path) as f:
	content = f.readlines()
	content = [i[:6] for i in content]
	print(content[0])
	plt.plot(content,color='m',linewidth=3,label='v1')
	plt.xlabel('epoch')
	plt.ylabel(path[:20])
	plt.xlim(0,101)
	#plt.ylim(0.5,1.0)
	'''
	x_major_locator=MultipleLocator(10)
	#把x轴的刻度间隔设置为1，并存在变量里
	y_major_locator=MultipleLocator(0.05)
	#把y轴的刻度间隔设置为10，并存在变量里
	ax=plt.gca()
	#ax为两条坐标轴的实例
	ax.xaxis.set_major_locator(x_major_locator)
	#把x轴的主刻度设置为1的倍数
	#ax.yaxis.set_major_locator(y_major_locator)
	#把y轴的主刻度设置为10的倍数
	#plt.ylim(0.0,1.0)
	#把x轴的刻度范围设置为-0.5到11，因为0.5不满一个刻度间隔，所以数字不会显示出来，但是能看到一点空白
	plt.xlim(0,101)
	'''
	plt.axis('auto') 
	

	plt.show()
'''
yy=[1,2,3,4,5,4,2,4,6,7]#随便创建了一个数据
xx=[3,5,4,1,2,3,4,5,6,3]
zz=[2,3,4,6,4,3,2,4,5,6]
plt.plot(yy,color='r',linewidth=5,linestyle=':',label='数据一')#color指定线条颜色，labeL标签内容
plt.plot(xx,color='g',linewidth=2,linestyle='--',label='数据二')#linewidth指定线条粗细
plt.plot(zz,color='b',linewidth=0.5,linestyle='-',label='数据三')#linestyle指定线形为点
plt.legend(loc=2)#标签展示位置，数字代表标签具位置
plt.xlabel('X轴称')
plt.ylabel('Y轴的名称')
plt.title('2018.7.30折线图示例')
plt.ylim(0,10)#Y轴标签范围为0-10
plt.show()
'''
