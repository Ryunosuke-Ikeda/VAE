from PIL import Image
import glob,os


files = glob.glob(os.path.join('./dataset_img/original_data/g_road/g_road/*.jpg'))


print(len(files))

output_dir='./RL/vae/dataset_img/crop_data/dataset/dataset1'

for f in files:
  try:
    image = Image.open(f)
    
  except OSError:
    print('Delete' + f)


  image_name=f.split('\\')[-1]
  image = image.resize((160,120))
  image.crop((0, 60, 160, 120)).save(f'{output_dir}/{image_name}', quality=95)

  