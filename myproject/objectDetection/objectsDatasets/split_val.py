import os, random, shutil

names = [x for x in os.listdir('./train/label/')]
val_names = random.sample(names, 65)
for name in val_names:
	shutil.move('./train/label/' + name, './val/label/' + name)
	shutil.move('./train/rgb/' + name[:-3]+'jpg', './val/rgb/' + name[:-3]+'jpg')
	shutil.move('./train/thermal/' + name[:-3]+'jpg', './val/thermal/' + name[:-3]+'jpg')
	# print('./train/label/' + name, './val/label/' + name)
	# print('./train/rgb/' + name[:-3]+'jpg', './val/rgb/' + name[:-3]+'jpg')
	# print('./train/thermal/' + name[:-3]+'jpg', './val/thermal/' + name[:-3]+'jpg')

