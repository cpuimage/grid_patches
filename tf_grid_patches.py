import tensorflow as tf
import matplotlib.pyplot as plt


def combined_static_and_dynamic_shape(x):
    static = x.get_shape().as_list()
    shape = tf.shape(x)
    ret = []
    for i, static_dim in enumerate(static):
        dim = static_dim or shape[i]
        ret.append(dim)
    return ret


def images_to_grid_patches(images, patch_size=(4, 4)):
    batch_size, height, width, channels = combined_static_and_dynamic_shape(images)
    patch_height, patch_width = int(height / patch_size[0]), int(height / patch_size[1])
    num_row = int(height / patch_height)
    num_patches = num_row * int(width / patch_width)
    out = tf.reshape(images, [batch_size, num_row, patch_height, width, channels])
    out = tf.transpose(out, [0, 1, 3, 2, 4])
    out = tf.reshape(out, [batch_size, num_patches, patch_width, patch_height, channels])
    patches = tf.transpose(out, [0, 1, 3, 2, 4])
    return patches


def grid_patches_to_images(patches, images_size=(256, 256)):
    height, width = images_size
    batch_size, num_patches, patch_height, patch_width, channels = combined_static_and_dynamic_shape(patches)
    num_row = int(height / patch_height)
    out = tf.transpose(patches, [0, 1, 3, 2, 4])
    out = tf.reshape(out, [batch_size, num_row, width, patch_height, channels])
    out = tf.transpose(out, [0, 1, 3, 2, 4])
    images = tf.reshape(out, [batch_size, height, width, channels])
    return images


def main():
    import tensorflow as tf
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    tf.compat.v1.enable_eager_execution()
    image_file = "demo.jpg"
    patch_size = (4, 4)
    images_size = (512, 512)
    apply_anything = False
    src_image = tf.cast(tf.image.decode_image(tf.io.read_file(image_file), channels=3), tf.float32) / 255.
    src_image = tf.image.resize(src_image, images_size)

    grid_patches = images_to_grid_patches(tf.expand_dims(src_image, 0), patch_size=patch_size)

    # apply anything in grid patches u want
    if apply_anything:
        grid_patches = tf.transpose(grid_patches, [0, 4, 2, 3, 1])
        grid_patches = tf.keras.layers.MaxPooling3D(pool_size=(3, 3, 3), strides=1,
                                                    padding="SAME")(grid_patches)
        grid_patches = tf.transpose(grid_patches, [0, 4, 2, 3, 1])

    dst_image = tf.squeeze(grid_patches_to_images(grid_patches, images_size=images_size), 0)
    plt.figure(figsize=(1 * (4 + 1), 5))
    plt.subplot(5, 1, 1)
    plt.imshow(dst_image)
    plt.title('outputs')
    plt.axis('off')
    for i, tile in enumerate(tf.squeeze(grid_patches, 0)):
        plt.subplot(5, 5, 5 + 1 + i)
        plt.imshow(tile)
        plt.title(str(i))
        plt.axis('off')
    plt.show()


if __name__ == '__main__':
    main()
