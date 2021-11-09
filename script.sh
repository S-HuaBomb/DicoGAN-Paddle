# Run a specific task by uncommenting corresponding line 

# evaluation
python DiscoGAN-Paddle/discogan/evaluation.py \
    --image_dir path/to/img_align_celeba \
    --attr_file path/to/list_attr_celeba_demo.txt \
    --n_test 10 \
    --task_name celebA \
    --style_A Male \
    --batch_size 1 \
    --ckpt_path path/to/discoGAN.pdparams
    --result_path ./results

# Run CelebA (eye_glass to no_eye_glass, only male)
python DiscoGAN-Paddle/discogan/image_translation.py \
    --image_dir path/to/img_align_celeba \
    --attr_file path/to/list_attr_celeba_demo.txt \
    --task_name celebA \
    --style_A Eyeglasses \
    --constraint Male --constraint_type 1 \

# Run CelebA Gender transform (male to female)
python ./discogan/image_translation.py \
    --image_dir path/to/img_align_celeba \
    --attr_file path/to/list_attr_celeba_demo.txt \
    --task_name celebA \
    --style_A Male \

# Run CelebA (blond to black hair, only female)
python ./discogan/image_translation.py \
    --image_dir path/to/img_align_celeba \
    --attr_file path/to/list_attr_celeba_demo.txt \
    --task_name celebA \
    --style_A Blond_Hair \
    --style_B Black_Hair \
    --constraint Male --constraint_type -1 \
