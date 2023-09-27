import data_gen.gt_generator as gt_gen

padding = 100

video_dir = "./datasets/test_dataset/video_source" #dataset_path
output_video_dir = "./datasets/test_dataset/classification_result"

test_data_loader = gt_gen.ReadVideoAsData(dataset_dir=video_dir, output_dir=output_video_dir)

test_data_loader._run()
