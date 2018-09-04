import json
import random
import os

Set = 'Goblins vs Gnomes'

# create a dict, keys are rarity, values are cardinfo.
def filterset():


	cards={'Common':[], 'Rare':[], 'Epic':[], 'Legendary':[]}

	f = open('/home/awesomejiang/scrapy/hs/cardinfo.json','r')

	for line in f:
		d =  json.loads(line)
		if Set in d['Set'] and 'Collectible' in d['Tip']:
			cards[ d['Rarity'][0] ].append(d)
		
	f.close()
	
	return cards


# open a single card, return [cardinfo,'Golden'/'Common']
def opensinglecard():
	cards = filterset()
	n = random.randint(1,100000)
	
	if n <111 :
		return [random.choice( cards['Legendary'] ) , 'Golden']
	elif n<1191 :
		return [random.choice( cards['Legendary'] ) , 'Common']
	elif n<1499 :
		return [random.choice( cards['Epic'] ) , 'Golden']
	elif n<5779 :
		return [random.choice( cards['Epic'] ) , 'Common']
	elif n<7149 :
		return [random.choice( cards['Rare'] ) , 'Golden']
	elif n<28549 :
		return [random.choice( cards['Rare'] ) , 'Common']
	elif n<30019 :
		return [random.choice( cards['Common'] ) , 'Golden']
	else :
		return [random.choice( cards['Common'] ) , 'Common']


#apply single card data to a packet,return[ [cardinfo,2/1]*5 ]
def openpacket():

	results = []
	
	counter = 0
	
	for x in range(5):
		card = opensinglecard()
		if 'Common' not in card[0]['Rarity']:
			counter += 1
		results.append(card)
		
	while counter == 0 :
		card = opensinglecard()
		if 'Common' not in card[0]['Rarity']:
			counter += 1
			results[0] = card
			
	return results





#test functions:

#output: card name, rarity, golden or not
def show():
	temp = openpacket()
	for x in temp:
		if x[1] == 'Golden':
			print( x[0]['Name'] + x[0]['Rarity'] + ['Golden'] )
		else:
			print( x[0]['Name'] + x[0]['Rarity'])



#sum the rarity of all cards 
def stat(num):

	d = {'Common': 0, 'Rare': 0, 'Epic': 0, 'Legendary': 0, 'Golden Common': 0, 'Golden Rare': 0, 'Golden Epic': 0, 'Golden Legendary': 0}
	
	for i in range(num):
		temp = openpacket()
		for x in temp :
			if x[1] == 'Golden':
				d['Golden'+' '+x[0]['Rarity'][0]] += 1
			else:
				d[x[0]['Rarity'][0]] += 1
				
	print(d)
		

#open packet with GUI:

from tkinter import *
from PIL import ImageTk

#find the pics
def grab_pics():

	cards = openpacket()
	imgs = []
	
	for x in cards:
		if x[1] == 'Golden':
			imgs.append( ImageTk.PhotoImage( file='/home/awesomejiang/scrapy/hs/'+x[0]['images'][1]['path'] ) )
		else:
			imgs.append(ImageTk.PhotoImage( file='/home/awesomejiang/scrapy/hs/'+x[0]['images'][0]['path'] ) )

	return imgs

#design a GUI

#cant show images (unsolved)
"""
class Openpacket_with_GUI(Frame):

	def __init__(self, master):
		Frame.__init__(self,master)
		self.pack()
		self.createwidgets()
		
	def createwidgets(self):
		imgs = grab_pics()
		for img in imgs:
			self.lb = Label(self,image=img).pack(side='right')
		
		self.btn = Button(self, text = 'try again').pack(side = 'bottom' )

		
root = Tk()
case = Openpacket_with_GUI(root)
root.mainloop()
"""


def Openpacket_with_GUI():
	root = Tk()

	imgs = grab_pics()
	for img in imgs:
		Label(root,image=img).pack(side='right')
			
	Button(root, text='close', command = quit).pack(side='right')

	root.mainloop()
	 

Openpacket_with_GUI()



















