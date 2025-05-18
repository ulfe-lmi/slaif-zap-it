# zap-it - Zero-shot Anything Pipeline for Image Tasks
High level computer vision (machine vision) pipeline composed of foundational models

Explanation: This is a high level pipeline for (almost) any real
computervision tasks. It is designed with two asumptions in mind.

1. There is infinite computing power available, you have big fat GPU with a
lot of VRAM, many CPU cores and infinite time (or patience). Let's say you
have a supercomputer - the main target for this code is HPC environment.

2. You don't want to program anything, you want to describe your problem to
a LLM model (e.g. chatgpt) give gim maybe an image or two of your problem
and ask him to generate yaml config file on your problem.

3. Then you run it and bitch to LLM (chatgpt) if config does not give you
good results, iteratively fixing the settings.

To summarize: you have HPC, infinite resources, and a problem to solve NOW.
It alignes well with the new paradigm of "AI factories" where supercomputers
should be used for various AI tasks. This aims to solve static image vision
problems. Or most of them.

Examples:

(sam2env) [jpers@wn212 zap-it]$ python zap-it-batch.py --config configs/glasswool.yaml --dir demos/glasswool/ --verbose full
Starting script...
[segment_images] Building SAM2 model...
[prepare_dirs] Created output folder: demos/glasswool/output
[CLIPFilter] loading clip-vit-base-patch32
Using a slow image processor as `use_fast` is unset and a slow processor was saved with this model. `use_fast=True` will be the default behavior in v4.52, even if the model was saved with a slow processor. This will result in minor differences in outputs. You'll still be able to use a slow processor with `use_fast=False`.

[process_folder] => Handling image: L_top_rectified.jpg
 => Original shape = 4056x3040
 => ROI=(0,1500,4000,1500) => partial shape=4000x1500
 => saved ROI debug => L_top_rectified-roi01.jpg
 => Single pass @native
[process_single_pass] Generating masks (single pass)...
[process_single_pass] => got 8 masks total.
[mask_generator debug] => saving raw SAM2 patches...
  => wrote L_top_rectified_sam2-patch0000.jpg
  => wrote L_top_rectified_sam2-patch0001.jpg
  => wrote L_top_rectified_sam2-patch0002.jpg
  => wrote L_top_rectified_sam2-patch0003.jpg
  => wrote L_top_rectified_sam2-patch0004.jpg
  => wrote L_top_rectified_sam2-patch0005.jpg
  => wrote L_top_rectified_sam2-patch0006.jpg
  => wrote L_top_rectified_sam2-patch0007.jpg
[postsam2processing] => from 8 => 7 remain by area/box
[clip_filter] => classifying 7 bounding boxes...
[CLIPFilter] mask=0, best_label='negative', score=0.2538, time=0.01s
[CLIPFilter debug] => wrote debug patch: L_top_rectified_patch0_a_blue_background.jpg
[CLIPFilter] mask=1, best_label='negative', score=0.2517, time=0.01s
[CLIPFilter debug] => wrote debug patch: L_top_rectified_patch1_a_blue_background.jpg
[CLIPFilter] mask=2, best_label='negative', score=0.2572, time=0.01s
[CLIPFilter debug] => wrote debug patch: L_top_rectified_patch2_a_blue_background.jpg
[CLIPFilter] mask=3, best_label='negative', score=0.2487, time=0.01s
[CLIPFilter debug] => wrote debug patch: L_top_rectified_patch3_a_blue_background.jpg
[CLIPFilter] mask=4, best_label='wool_surface', score=0.3579, time=0.03s
[CLIPFilter debug] => wrote debug patch: L_top_rectified_patch4_the_textured_front_of_a_rock_wool_substrate.jpg
[CLIPFilter] mask=5, best_label='negative', score=0.2519, time=0.01s
[CLIPFilter debug] => wrote debug patch: L_top_rectified_patch5_a_blue_background.jpg
[CLIPFilter] mask=6, best_label='negative', score=0.2566, time=0.02s
[CLIPFilter debug] => wrote debug patch: L_top_rectified_patch6_a_blue_background.jpg
[clip_filter] => classification done, now final label filter...
[postsam2processing debug] => saving final patches after classification...
  => wrote final patch => L_top_rectified_sam2-filtered-patch0000.jpg
[visualization] => building 'pre' 2x2 composite (sam2) ...
  => [build_composite_for_masks] building annotated overlay...
  => [build_composite_for_masks] building random color + masked array...
  => [build_composite_for_masks] building 2x2 now...
[visualization] => building 'post' 2x2 composite (clip) ...
  => [build_composite_for_masks] building annotated overlay...
  => [build_composite_for_masks] building random color + masked array...
  => [build_composite_for_masks] building 2x2 now...
[visualization] => generating panoptic final image with detectron2 Visualizer...
[visualization] => wrote final single overlay => demos/glasswool/output/L_top_rectified-final.jpg
[geometry] => geometry section found => applying canny/hough to each final mask... (#masks=1)
[geometry] => applying canny on mask 0, shape=(3040, 4056), thr=(50,150), aperture=3
[geometry] => wrote geometry overlay => demos/glasswool/output/L_top_rectified_geometry.jpg
[process_folder] => wrote JSON => demos/glasswool/output/L_top_rectified.json
[visualization] => wrote summary => demos/glasswool/output/L_top_rectified_summary.jpg
[process_folder] => done with image.



(sam2env) [jpers@wn212 zap-it]$ python zap-it-batch.py --config configs/goats.yaml --dir demos/goats/ --verbose full
Starting script...
[segment_images] Building SAM2 model...
[prepare_dirs] Created output folder: demos/goats/output
[CLIPFilter] loading clip-vit-base-patch32
Using a slow image processor as `use_fast` is unset and a slow processor was saved with this model. `use_fast=True` will be the default behavior in v4.52, even if the model was saved with a slow processor. This will result in minor differences in outputs. You'll still be able to use a slow processor with `use_fast=False`.

[process_folder] => Handling image: goats1.jpg
 => Original shape = 5568x4176
 => ROI=(5,1825,5348,1092) => partial shape=5348x1092
 => saved ROI debug => goats1-roi01.jpg
 => Single pass @native
[process_single_pass] Generating masks (single pass)...
[process_single_pass] => got 106 masks total.
[mask_generator debug] => saving raw SAM2 patches...
  => wrote goats1_sam2-patch0000.jpg
  => wrote goats1_sam2-patch0001.jpg
  => wrote goats1_sam2-patch0002.jpg
  => wrote goats1_sam2-patch0003.jpg
  => wrote goats1_sam2-patch0004.jpg
  => wrote goats1_sam2-patch0005.jpg
  => wrote goats1_sam2-patch0006.jpg
  => wrote goats1_sam2-patch0007.jpg
  => wrote goats1_sam2-patch0008.jpg
  => wrote goats1_sam2-patch0009.jpg
  => wrote goats1_sam2-patch0010.jpg
  => wrote goats1_sam2-patch0011.jpg
  => wrote goats1_sam2-patch0012.jpg
  => wrote goats1_sam2-patch0013.jpg
  => wrote goats1_sam2-patch0014.jpg
  => wrote goats1_sam2-patch0015.jpg
  => wrote goats1_sam2-patch0016.jpg
  => wrote goats1_sam2-patch0017.jpg
  => wrote goats1_sam2-patch0018.jpg
  => wrote goats1_sam2-patch0019.jpg
  => wrote goats1_sam2-patch0020.jpg
  => wrote goats1_sam2-patch0021.jpg
  => wrote goats1_sam2-patch0022.jpg
  => wrote goats1_sam2-patch0023.jpg
  => wrote goats1_sam2-patch0024.jpg
  => wrote goats1_sam2-patch0025.jpg
  => wrote goats1_sam2-patch0026.jpg
  => wrote goats1_sam2-patch0027.jpg
  => wrote goats1_sam2-patch0028.jpg
  => wrote goats1_sam2-patch0029.jpg
  => wrote goats1_sam2-patch0030.jpg
  => wrote goats1_sam2-patch0031.jpg
  => wrote goats1_sam2-patch0032.jpg
  => wrote goats1_sam2-patch0033.jpg
  => wrote goats1_sam2-patch0034.jpg
  => wrote goats1_sam2-patch0035.jpg
  => wrote goats1_sam2-patch0036.jpg
  => wrote goats1_sam2-patch0037.jpg
  => wrote goats1_sam2-patch0038.jpg
  => wrote goats1_sam2-patch0039.jpg
  => wrote goats1_sam2-patch0040.jpg
  => wrote goats1_sam2-patch0041.jpg
  => wrote goats1_sam2-patch0042.jpg
  => wrote goats1_sam2-patch0043.jpg
  => wrote goats1_sam2-patch0044.jpg
  => wrote goats1_sam2-patch0045.jpg
  => wrote goats1_sam2-patch0046.jpg
  => wrote goats1_sam2-patch0047.jpg
  => wrote goats1_sam2-patch0048.jpg
  => wrote goats1_sam2-patch0049.jpg
  => wrote goats1_sam2-patch0050.jpg
  => wrote goats1_sam2-patch0051.jpg
  => wrote goats1_sam2-patch0052.jpg
  => wrote goats1_sam2-patch0053.jpg
  => wrote goats1_sam2-patch0054.jpg
  => wrote goats1_sam2-patch0055.jpg
  => wrote goats1_sam2-patch0056.jpg
  => wrote goats1_sam2-patch0057.jpg
  => wrote goats1_sam2-patch0058.jpg
  => wrote goats1_sam2-patch0059.jpg
  => wrote goats1_sam2-patch0060.jpg
  => wrote goats1_sam2-patch0061.jpg
  => wrote goats1_sam2-patch0062.jpg
  => wrote goats1_sam2-patch0063.jpg
  => wrote goats1_sam2-patch0064.jpg
  => wrote goats1_sam2-patch0065.jpg
  => wrote goats1_sam2-patch0066.jpg
  => wrote goats1_sam2-patch0067.jpg
  => wrote goats1_sam2-patch0068.jpg
  => wrote goats1_sam2-patch0069.jpg
  => wrote goats1_sam2-patch0070.jpg
  => wrote goats1_sam2-patch0071.jpg
  => wrote goats1_sam2-patch0072.jpg
  => wrote goats1_sam2-patch0073.jpg
  => wrote goats1_sam2-patch0074.jpg
  => wrote goats1_sam2-patch0075.jpg
  => wrote goats1_sam2-patch0076.jpg
  => wrote goats1_sam2-patch0077.jpg
  => wrote goats1_sam2-patch0078.jpg
  => wrote goats1_sam2-patch0079.jpg
  => wrote goats1_sam2-patch0080.jpg
  => wrote goats1_sam2-patch0081.jpg
  => wrote goats1_sam2-patch0082.jpg
  => wrote goats1_sam2-patch0083.jpg
  => wrote goats1_sam2-patch0084.jpg
  => wrote goats1_sam2-patch0085.jpg
  => wrote goats1_sam2-patch0086.jpg
  => wrote goats1_sam2-patch0087.jpg
  => wrote goats1_sam2-patch0088.jpg
  => wrote goats1_sam2-patch0089.jpg
  => wrote goats1_sam2-patch0090.jpg
  => wrote goats1_sam2-patch0091.jpg
  => wrote goats1_sam2-patch0092.jpg
  => wrote goats1_sam2-patch0093.jpg
  => wrote goats1_sam2-patch0094.jpg
  => wrote goats1_sam2-patch0095.jpg
  => wrote goats1_sam2-patch0096.jpg
  => wrote goats1_sam2-patch0097.jpg
  => wrote goats1_sam2-patch0098.jpg
  => wrote goats1_sam2-patch0099.jpg
  => wrote goats1_sam2-patch0100.jpg
  => wrote goats1_sam2-patch0101.jpg
  => wrote goats1_sam2-patch0102.jpg
  => wrote goats1_sam2-patch0103.jpg
  => wrote goats1_sam2-patch0104.jpg
  => wrote goats1_sam2-patch0105.jpg
[postsam2processing] => from 106 => 81 remain by area/box
[clip_filter] => classifying 81 bounding boxes...
[CLIPFilter] mask=0, best_label='negative', score=0.3270, time=0.01s
[CLIPFilter debug] => wrote debug patch: goats1_patch0_a_section_of_grass_with_white_dots.jpg
[CLIPFilter] mask=1, best_label='negative', score=0.2809, time=0.01s
[CLIPFilter debug] => wrote debug patch: goats1_patch1_a_section_of_grass_with_white_dots.jpg
[CLIPFilter] mask=2, best_label='sign', score=0.3542, time=0.01s
[CLIPFilter debug] => wrote debug patch: goats1_patch2_a_white_tag_showing_an_agricultural_zone_number.jpg
[CLIPFilter] mask=3, best_label='sign', score=0.3389, time=0.01s
[CLIPFilter debug] => wrote debug patch: goats1_patch3_a_white_tag_showing_an_agricultural_zone_number.jpg
[CLIPFilter] mask=4, best_label='sign', score=0.3505, time=0.01s
[CLIPFilter debug] => wrote debug patch: goats1_patch4_a_printed_sign_showing_a_field_number.jpg
[CLIPFilter] mask=5, best_label='sign', score=0.2745, time=0.01s
[CLIPFilter debug] => wrote debug patch: goats1_patch5_a_grassy_field_tag_with_black_digits.jpg
[CLIPFilter] mask=6, best_label='sign', score=0.2924, time=0.01s
[CLIPFilter debug] => wrote debug patch: goats1_patch6_a_numbered_placard_staked_in_soil.jpg
[CLIPFilter] mask=7, best_label='sign', score=0.3377, time=0.01s
[CLIPFilter debug] => wrote debug patch: goats1_patch7_a_numeric_sign_in_an_agricultural_test_zone.jpg
[CLIPFilter] mask=8, best_label='sign', score=0.3451, time=0.01s
[CLIPFilter debug] => wrote debug patch: goats1_patch8_a_pole-mounted_sign_with_a_test_plot_number.jpg
[CLIPFilter] mask=9, best_label='sign', score=0.3404, time=0.01s
[CLIPFilter debug] => wrote debug patch: goats1_patch9_a_labeled_plot_sign_on_a_post.jpg
[CLIPFilter] mask=10, best_label='negative', score=0.2951, time=0.01s
[CLIPFilter debug] => wrote debug patch: goats1_patch10_a_group_of_white_stems_with_no_sign.jpg
[CLIPFilter] mask=11, best_label='negative', score=0.2871, time=0.01s
[CLIPFilter debug] => wrote debug patch: goats1_patch11_a_grass_shadow_looking_like_a_sign.jpg
[CLIPFilter] mask=12, best_label='sign', score=0.3488, time=0.01s
[CLIPFilter debug] => wrote debug patch: goats1_patch12_a_pole-mounted_sign_with_a_test_plot_number.jpg
[CLIPFilter] mask=13, best_label='negative', score=0.3024, time=0.01s
[CLIPFilter debug] => wrote debug patch: goats1_patch13_a_white_clover_patch.jpg
[CLIPFilter] mask=14, best_label='sign', score=0.3629, time=0.01s
[CLIPFilter debug] => wrote debug patch: goats1_patch14_a_pole-mounted_sign_with_a_test_plot_number.jpg
[CLIPFilter] mask=15, best_label='sign', score=0.3567, time=0.01s
[CLIPFilter debug] => wrote debug patch: goats1_patch15_a_white_tag_showing_an_agricultural_zone_number.jpg
[CLIPFilter] mask=16, best_label='negative', score=0.2993, time=0.01s
[CLIPFilter debug] => wrote debug patch: goats1_patch16_a_blurry_object_between_wires.jpg
[CLIPFilter] mask=17, best_label='sign', score=0.3142, time=0.01s
[CLIPFilter debug] => wrote debug patch: goats1_patch17_a_white_sign_showing_a_number.jpg
[CLIPFilter] mask=18, best_label='negative', score=0.2906, time=0.01s
[CLIPFilter debug] => wrote debug patch: goats1_patch18_a_dark_vertical_object_in_the_ground.jpg
[CLIPFilter] mask=19, best_label='sign', score=0.3024, time=0.01s
[CLIPFilter debug] => wrote debug patch: goats1_patch19_a_labeled_field_marker_in_experimental_grass.jpg
[CLIPFilter] mask=20, best_label='sign', score=0.2778, time=0.01s
[CLIPFilter debug] => wrote debug patch: goats1_patch20_a_numbered_white_rectangle_in_tall_grass.jpg
[CLIPFilter] mask=21, best_label='negative', score=0.3025, time=0.01s
[CLIPFilter debug] => wrote debug patch: goats1_patch21_white_petals_scattered_on_grass.jpg
[CLIPFilter] mask=22, best_label='negative', score=0.2961, time=0.01s
[CLIPFilter debug] => wrote debug patch: goats1_patch22_white_petals_scattered_on_grass.jpg
[CLIPFilter] mask=23, best_label='goat', score=0.3574, time=0.01s
[CLIPFilter debug] => wrote debug patch: goats1_patch23_a_brown_and_white_goat_in_the_middle_of_the_pasture.jpg
[CLIPFilter] mask=24, best_label='goat', score=0.3367, time=0.01s
[CLIPFilter debug] => wrote debug patch: goats1_patch24_a_goat_grazing_between_numbered_posts.jpg
[CLIPFilter] mask=25, best_label='sign', score=0.3445, time=0.01s
[CLIPFilter debug] => wrote debug patch: goats1_patch25_a_white_sign_showing_a_number.jpg
[CLIPFilter] mask=26, best_label='negative', score=0.2780, time=0.01s
[CLIPFilter debug] => wrote debug patch: goats1_patch26_a_grass_shadow_looking_like_a_sign.jpg
[CLIPFilter] mask=27, best_label='sign', score=0.2886, time=0.01s
[CLIPFilter debug] => wrote debug patch: goats1_patch27_a_weathered_sign_with_a_legible_number.jpg
[CLIPFilter] mask=28, best_label='sign', score=0.2915, time=0.01s
[CLIPFilter debug] => wrote debug patch: goats1_patch28_a_grassy_field_tag_with_black_digits.jpg
[CLIPFilter] mask=29, best_label='sign', score=0.2941, time=0.01s
[CLIPFilter debug] => wrote debug patch: goats1_patch29_a_plastic_card_label_in_a_fenced_pasture.jpg
[CLIPFilter] mask=30, best_label='sign', score=0.3341, time=0.01s
[CLIPFilter debug] => wrote debug patch: goats1_patch30_a_numeric_sign_in_an_agricultural_test_zone.jpg
[CLIPFilter] mask=31, best_label='negative', score=0.3221, time=0.01s
[CLIPFilter debug] => wrote debug patch: goats1_patch31_white_petals_scattered_on_grass.jpg
[CLIPFilter] mask=32, best_label='negative', score=0.2991, time=0.01s
[CLIPFilter debug] => wrote debug patch: goats1_patch32_a_pattern_of_lines_in_green_weeds.jpg
[CLIPFilter] mask=33, best_label='negative', score=0.3079, time=0.01s
[CLIPFilter debug] => wrote debug patch: goats1_patch33_white_petals_scattered_on_grass.jpg
[CLIPFilter] mask=34, best_label='negative', score=0.3283, time=0.01s
[CLIPFilter debug] => wrote debug patch: goats1_patch34_a_section_of_grass_with_white_dots.jpg
[CLIPFilter] mask=35, best_label='negative', score=0.3182, time=0.01s
[CLIPFilter debug] => wrote debug patch: goats1_patch35_a_bunch_of_wild_daisies.jpg
[CLIPFilter] mask=36, best_label='goat', score=0.3392, time=0.01s
[CLIPFilter debug] => wrote debug patch: goats1_patch36_a_white_and_brown_goat_seen_from_a_distance.jpg
[CLIPFilter] mask=37, best_label='goat', score=0.3584, time=0.01s
[CLIPFilter debug] => wrote debug patch: goats1_patch37_a_brown_and_white_goat_in_the_middle_of_the_pasture.jpg
[CLIPFilter] mask=38, best_label='sign', score=0.2883, time=0.01s
[CLIPFilter debug] => wrote debug patch: goats1_patch38_a_numbered_white_rectangle_in_tall_grass.jpg
[CLIPFilter] mask=39, best_label='sign', score=0.3146, time=0.01s
[CLIPFilter debug] => wrote debug patch: goats1_patch39_a_white_sign_showing_a_number.jpg
[CLIPFilter] mask=40, best_label='sign', score=0.3373, time=0.01s
[CLIPFilter debug] => wrote debug patch: goats1_patch40_a_printed_sign_showing_a_field_number.jpg
[CLIPFilter] mask=41, best_label='goat', score=0.3266, time=0.01s
[CLIPFilter debug] => wrote debug patch: goats1_patch41_a_muscular_white_goat_with_brown_markings.jpg
[CLIPFilter] mask=42, best_label='goat', score=0.3167, time=0.01s
[CLIPFilter debug] => wrote debug patch: goats1_patch42_a_Boer_goat_partially_hidden_by_tall_grass.jpg
[CLIPFilter] mask=43, best_label='goat', score=0.3317, time=0.01s
[CLIPFilter debug] => wrote debug patch: goats1_patch43_a_Boer_goat_eating_plants_in_a_test_plot.jpg
[CLIPFilter] mask=44, best_label='goat', score=0.3291, time=0.01s
[CLIPFilter debug] => wrote debug patch: goats1_patch44_a_Boer_goat_partially_hidden_by_tall_grass.jpg
[CLIPFilter] mask=45, best_label='sign', score=0.3555, time=0.01s
[CLIPFilter debug] => wrote debug patch: goats1_patch45_a_white_laminated_plot_number_sign.jpg
[CLIPFilter] mask=46, best_label='sign', score=0.3680, time=0.01s
[CLIPFilter debug] => wrote debug patch: goats1_patch46_a_pole-mounted_sign_with_a_test_plot_number.jpg
[CLIPFilter] mask=47, best_label='sign', score=0.3566, time=0.01s
[CLIPFilter debug] => wrote debug patch: goats1_patch47_a_numeric_sign_in_an_agricultural_test_zone.jpg
[CLIPFilter] mask=48, best_label='sign', score=0.3350, time=0.01s
[CLIPFilter debug] => wrote debug patch: goats1_patch48_a_printed_sign_showing_a_field_number.jpg
[CLIPFilter] mask=49, best_label='sign', score=0.3348, time=0.01s
[CLIPFilter debug] => wrote debug patch: goats1_patch49_a_printed_sign_showing_a_field_number.jpg
[CLIPFilter] mask=50, best_label='sign', score=0.3463, time=0.01s
[CLIPFilter debug] => wrote debug patch: goats1_patch50_a_white_tag_showing_an_agricultural_zone_number.jpg
[CLIPFilter] mask=51, best_label='sign', score=0.3715, time=0.01s
[CLIPFilter debug] => wrote debug patch: goats1_patch51_a_white_tag_showing_an_agricultural_zone_number.jpg
[CLIPFilter] mask=52, best_label='negative', score=0.3279, time=0.01s
[CLIPFilter debug] => wrote debug patch: goats1_patch52_white_petals_scattered_on_grass.jpg
[CLIPFilter] mask=53, best_label='sign', score=0.3756, time=0.01s
[CLIPFilter debug] => wrote debug patch: goats1_patch53_a_pole-mounted_sign_with_a_test_plot_number.jpg
[CLIPFilter] mask=54, best_label='sign', score=0.3541, time=0.01s
[CLIPFilter debug] => wrote debug patch: goats1_patch54_a_white_sign_showing_a_number.jpg
[CLIPFilter] mask=55, best_label='negative', score=0.3009, time=0.01s
[CLIPFilter debug] => wrote debug patch: goats1_patch55_a_section_of_grass_with_white_dots.jpg
[CLIPFilter] mask=56, best_label='sign', score=0.3105, time=0.01s
[CLIPFilter debug] => wrote debug patch: goats1_patch56_a_numbered_white_rectangle_in_tall_grass.jpg
[CLIPFilter] mask=57, best_label='negative', score=0.2974, time=0.01s
[CLIPFilter debug] => wrote debug patch: goats1_patch57_a_section_of_grass_with_white_dots.jpg
[CLIPFilter] mask=58, best_label='sign', score=0.2764, time=0.01s
[CLIPFilter debug] => wrote debug patch: goats1_patch58_a_white_ID_sign_above_the_grass_line.jpg
[CLIPFilter] mask=59, best_label='negative', score=0.2939, time=0.01s
[CLIPFilter debug] => wrote debug patch: goats1_patch59_a_pattern_of_lines_in_green_weeds.jpg
[CLIPFilter] mask=60, best_label='goat', score=0.2803, time=0.01s
[CLIPFilter debug] => wrote debug patch: goats1_patch60_a_goat’s_back_visible_in_tall_grass.jpg
[CLIPFilter] mask=61, best_label='negative', score=0.3046, time=0.01s
[CLIPFilter debug] => wrote debug patch: goats1_patch61_a_wooden_stake_with_no_marking.jpg
[CLIPFilter] mask=62, best_label='negative', score=0.3048, time=0.01s
[CLIPFilter debug] => wrote debug patch: goats1_patch62_a_section_of_grass_with_white_dots.jpg
[CLIPFilter] mask=63, best_label='negative', score=0.2897, time=0.01s
[CLIPFilter debug] => wrote debug patch: goats1_patch63_white_petals_scattered_on_grass.jpg
[CLIPFilter] mask=64, best_label='sign', score=0.3483, time=0.01s
[CLIPFilter debug] => wrote debug patch: goats1_patch64_a_white_tag_showing_an_agricultural_zone_number.jpg
[CLIPFilter] mask=65, best_label='negative', score=0.2854, time=0.01s
[CLIPFilter debug] => wrote debug patch: goats1_patch65_a_red_marker_flag.jpg
[CLIPFilter] mask=66, best_label='goat', score=0.3082, time=0.01s
[CLIPFilter debug] => wrote debug patch: goats1_patch66_a_white_and_brown_goat_seen_from_a_distance.jpg
[CLIPFilter] mask=67, best_label='sign', score=0.3459, time=0.01s
[CLIPFilter debug] => wrote debug patch: goats1_patch67_a_white_laminated_plot_number_sign.jpg
[CLIPFilter] mask=68, best_label='goat', score=0.3401, time=0.01s
[CLIPFilter debug] => wrote debug patch: goats1_patch68_a_white_and_brown_goat_seen_from_a_distance.jpg
[CLIPFilter] mask=69, best_label='sign', score=0.3377, time=0.01s
[CLIPFilter debug] => wrote debug patch: goats1_patch69_a_white_tag_showing_an_agricultural_zone_number.jpg
[CLIPFilter] mask=70, best_label='sign', score=0.3576, time=0.01s
[CLIPFilter debug] => wrote debug patch: goats1_patch70_a_white_tag_showing_an_agricultural_zone_number.jpg
[CLIPFilter] mask=71, best_label='sign', score=0.3538, time=0.01s
[CLIPFilter debug] => wrote debug patch: goats1_patch71_a_numeric_sign_in_an_agricultural_test_zone.jpg
[CLIPFilter] mask=72, best_label='negative', score=0.2993, time=0.01s
[CLIPFilter debug] => wrote debug patch: goats1_patch72_a_soil_patch_among_overgrown_weeds.jpg
[CLIPFilter] mask=73, best_label='goat', score=0.3561, time=0.01s
[CLIPFilter debug] => wrote debug patch: goats1_patch73_a_white_and_brown_goat_seen_from_a_distance.jpg
[CLIPFilter] mask=74, best_label='goat', score=0.2996, time=0.01s
[CLIPFilter debug] => wrote debug patch: goats1_patch74_a_Boer_goat_eating_plants_in_a_test_plot.jpg
[CLIPFilter] mask=75, best_label='negative', score=0.3290, time=0.01s
[CLIPFilter debug] => wrote debug patch: goats1_patch75_a_group_of_light-colored_flowers.jpg
[CLIPFilter] mask=76, best_label='sign', score=0.3780, time=0.01s
[CLIPFilter debug] => wrote debug patch: goats1_patch76_a_pole-mounted_sign_with_a_test_plot_number.jpg
[CLIPFilter] mask=77, best_label='sign', score=0.2911, time=0.01s
[CLIPFilter debug] => wrote debug patch: goats1_patch77_a_numbered_white_rectangle_in_tall_grass.jpg
[CLIPFilter] mask=78, best_label='negative', score=0.2674, time=0.01s
[CLIPFilter debug] => wrote debug patch: goats1_patch78_a_darkened_green_square.jpg
[CLIPFilter] mask=79, best_label='sign', score=0.3541, time=0.01s
[CLIPFilter debug] => wrote debug patch: goats1_patch79_a_pole-mounted_sign_with_a_test_plot_number.jpg
[CLIPFilter] mask=80, best_label='sign', score=0.2716, time=0.01s
[CLIPFilter debug] => wrote debug patch: goats1_patch80_a_grassy_field_tag_with_black_digits.jpg
[clip_filter] => classification done, now final label filter...
[postsam2processing debug] => saving final patches after classification...
  => wrote final patch => goats1_sam2-filtered-patch0000.jpg
  => wrote final patch => goats1_sam2-filtered-patch0001.jpg
  => wrote final patch => goats1_sam2-filtered-patch0002.jpg
  => wrote final patch => goats1_sam2-filtered-patch0003.jpg
  => wrote final patch => goats1_sam2-filtered-patch0004.jpg
  => wrote final patch => goats1_sam2-filtered-patch0005.jpg
  => wrote final patch => goats1_sam2-filtered-patch0006.jpg
  => wrote final patch => goats1_sam2-filtered-patch0007.jpg
  => wrote final patch => goats1_sam2-filtered-patch0008.jpg
  => wrote final patch => goats1_sam2-filtered-patch0009.jpg
  => wrote final patch => goats1_sam2-filtered-patch0010.jpg
  => wrote final patch => goats1_sam2-filtered-patch0011.jpg
  => wrote final patch => goats1_sam2-filtered-patch0012.jpg
[visualization] => building 'pre' 2x2 composite (sam2) ...
  => [build_composite_for_masks] building annotated overlay...
  => [build_composite_for_masks] building random color + masked array...
  => [build_composite_for_masks] building 2x2 now...
[visualization] => building 'post' 2x2 composite (clip) ...
  => [build_composite_for_masks] building annotated overlay...
  => [build_composite_for_masks] building random color + masked array...
  => [build_composite_for_masks] building 2x2 now...
[visualization] => generating panoptic final image with detectron2 Visualizer...
[visualization] => wrote final single overlay => demos/goats/output/goats1-final.jpg
[process_folder] => wrote JSON => demos/goats/output/goats1.json
[visualization] => wrote summary => demos/goats/output/goats1_summary.jpg
[process_folder] => done with image.


[process_folder] => Handling image: goats2.jpg
 => Original shape = 5568x4176
 => ROI=(5,1825,5348,1092) => partial shape=5348x1092
 => saved ROI debug => goats2-roi01.jpg
 => Single pass @native
[process_single_pass] Generating masks (single pass)...
[process_single_pass] => got 79 masks total.
[mask_generator debug] => saving raw SAM2 patches...
  => wrote goats2_sam2-patch0000.jpg
  => wrote goats2_sam2-patch0001.jpg
  => wrote goats2_sam2-patch0002.jpg
  => wrote goats2_sam2-patch0003.jpg
  => wrote goats2_sam2-patch0004.jpg
  => wrote goats2_sam2-patch0005.jpg
  => wrote goats2_sam2-patch0006.jpg
  => wrote goats2_sam2-patch0007.jpg
  => wrote goats2_sam2-patch0008.jpg
  => wrote goats2_sam2-patch0009.jpg
  => wrote goats2_sam2-patch0010.jpg
  => wrote goats2_sam2-patch0011.jpg
  => wrote goats2_sam2-patch0012.jpg
  => wrote goats2_sam2-patch0013.jpg
  => wrote goats2_sam2-patch0014.jpg
  => wrote goats2_sam2-patch0015.jpg
  => wrote goats2_sam2-patch0016.jpg
  => wrote goats2_sam2-patch0017.jpg
  => wrote goats2_sam2-patch0018.jpg
  => wrote goats2_sam2-patch0019.jpg
  => wrote goats2_sam2-patch0020.jpg
  => wrote goats2_sam2-patch0021.jpg
  => wrote goats2_sam2-patch0022.jpg
  => wrote goats2_sam2-patch0023.jpg
  => wrote goats2_sam2-patch0024.jpg
  => wrote goats2_sam2-patch0025.jpg
  => wrote goats2_sam2-patch0026.jpg
  => wrote goats2_sam2-patch0027.jpg
  => wrote goats2_sam2-patch0028.jpg
  => wrote goats2_sam2-patch0029.jpg
  => wrote goats2_sam2-patch0030.jpg
  => wrote goats2_sam2-patch0031.jpg
  => wrote goats2_sam2-patch0032.jpg
  => wrote goats2_sam2-patch0033.jpg
  => wrote goats2_sam2-patch0034.jpg
  => wrote goats2_sam2-patch0035.jpg
  => wrote goats2_sam2-patch0036.jpg
  => wrote goats2_sam2-patch0037.jpg
  => wrote goats2_sam2-patch0038.jpg
  => wrote goats2_sam2-patch0039.jpg
  => wrote goats2_sam2-patch0040.jpg
  => wrote goats2_sam2-patch0041.jpg
  => wrote goats2_sam2-patch0042.jpg
  => wrote goats2_sam2-patch0043.jpg
  => wrote goats2_sam2-patch0044.jpg
  => wrote goats2_sam2-patch0045.jpg
  => wrote goats2_sam2-patch0046.jpg
  => wrote goats2_sam2-patch0047.jpg
  => wrote goats2_sam2-patch0048.jpg
  => wrote goats2_sam2-patch0049.jpg
  => wrote goats2_sam2-patch0050.jpg
  => wrote goats2_sam2-patch0051.jpg
  => wrote goats2_sam2-patch0052.jpg
  => wrote goats2_sam2-patch0053.jpg
  => wrote goats2_sam2-patch0054.jpg
  => wrote goats2_sam2-patch0055.jpg
  => wrote goats2_sam2-patch0056.jpg
  => wrote goats2_sam2-patch0057.jpg
  => wrote goats2_sam2-patch0058.jpg
  => wrote goats2_sam2-patch0059.jpg
  => wrote goats2_sam2-patch0060.jpg
  => wrote goats2_sam2-patch0061.jpg
  => wrote goats2_sam2-patch0062.jpg
  => wrote goats2_sam2-patch0063.jpg
  => wrote goats2_sam2-patch0064.jpg
  => wrote goats2_sam2-patch0065.jpg
  => wrote goats2_sam2-patch0066.jpg
  => wrote goats2_sam2-patch0067.jpg
  => wrote goats2_sam2-patch0068.jpg
  => wrote goats2_sam2-patch0069.jpg
  => wrote goats2_sam2-patch0070.jpg
  => wrote goats2_sam2-patch0071.jpg
  => wrote goats2_sam2-patch0072.jpg
  => wrote goats2_sam2-patch0073.jpg
  => wrote goats2_sam2-patch0074.jpg
  => wrote goats2_sam2-patch0075.jpg
  => wrote goats2_sam2-patch0076.jpg
  => wrote goats2_sam2-patch0077.jpg
  => wrote goats2_sam2-patch0078.jpg
[postsam2processing] => from 79 => 59 remain by area/box
[clip_filter] => classifying 59 bounding boxes...
[CLIPFilter] mask=0, best_label='negative', score=0.2801, time=0.01s
[CLIPFilter debug] => wrote debug patch: goats2_patch0_a_small_branch_shaped_like_a_tag.jpg
[CLIPFilter] mask=1, best_label='negative', score=0.2792, time=0.01s
[CLIPFilter debug] => wrote debug patch: goats2_patch1_a_section_of_grass_with_white_dots.jpg
[CLIPFilter] mask=2, best_label='negative', score=0.2947, time=0.01s
[CLIPFilter debug] => wrote debug patch: goats2_patch2_a_section_of_grass_with_white_dots.jpg
[CLIPFilter] mask=3, best_label='negative', score=0.2841, time=0.01s
[CLIPFilter debug] => wrote debug patch: goats2_patch3_a_pale_patch_in_the_turf.jpg
[CLIPFilter] mask=4, best_label='negative', score=0.3084, time=0.01s
[CLIPFilter debug] => wrote debug patch: goats2_patch4_a_section_of_grass_with_white_dots.jpg
[CLIPFilter] mask=5, best_label='negative', score=0.3003, time=0.01s
[CLIPFilter debug] => wrote debug patch: goats2_patch5_a_tree_branch_lying_on_pasture.jpg
[CLIPFilter] mask=6, best_label='negative', score=0.3003, time=0.01s
[CLIPFilter debug] => wrote debug patch: goats2_patch6_a_blurry_corner_with_grass_glare.jpg
[CLIPFilter] mask=7, best_label='goat', score=0.3150, time=0.01s
[CLIPFilter debug] => wrote debug patch: goats2_patch7_a_white-bodied_goat_on_green_grass.jpg
[CLIPFilter] mask=8, best_label='negative', score=0.2795, time=0.01s
[CLIPFilter debug] => wrote debug patch: goats2_patch8_a_dark_vertical_object_in_the_ground.jpg
[CLIPFilter] mask=9, best_label='sign', score=0.2931, time=0.01s
[CLIPFilter debug] => wrote debug patch: goats2_patch9_a_white_ID_sign_above_the_grass_line.jpg
[CLIPFilter] mask=10, best_label='goat', score=0.2892, time=0.01s
[CLIPFilter debug] => wrote debug patch: goats2_patch10_a_white_and_brown_goat_seen_from_a_distance.jpg
[CLIPFilter] mask=11, best_label='goat', score=0.3492, time=0.01s
[CLIPFilter debug] => wrote debug patch: goats2_patch11_a_brown_and_white_goat_in_the_middle_of_the_pasture.jpg
[CLIPFilter] mask=12, best_label='sign', score=0.3584, time=0.01s
[CLIPFilter debug] => wrote debug patch: goats2_patch12_a_square_white_field_marker_with_digits.jpg
[CLIPFilter] mask=13, best_label='goat', score=0.3410, time=0.01s
[CLIPFilter debug] => wrote debug patch: goats2_patch13_a_white_and_brown_goat_seen_from_a_distance.jpg
[CLIPFilter] mask=14, best_label='sign', score=0.3489, time=0.01s
[CLIPFilter debug] => wrote debug patch: goats2_patch14_a_white_laminated_plot_number_sign.jpg
[CLIPFilter] mask=15, best_label='negative', score=0.2629, time=0.01s
[CLIPFilter debug] => wrote debug patch: goats2_patch15_a_dark_vertical_object_in_the_ground.jpg
[CLIPFilter] mask=16, best_label='sign', score=0.2854, time=0.01s
[CLIPFilter debug] => wrote debug patch: goats2_patch16_a_numbered_white_rectangle_in_tall_grass.jpg
[CLIPFilter] mask=17, best_label='sign', score=0.3045, time=0.01s
[CLIPFilter debug] => wrote debug patch: goats2_patch17_a_numbered_white_rectangle_in_tall_grass.jpg
[CLIPFilter] mask=18, best_label='goat', score=0.3200, time=0.01s
[CLIPFilter debug] => wrote debug patch: goats2_patch18_a_white_and_brown_goat_seen_from_a_distance.jpg
[CLIPFilter] mask=19, best_label='goat', score=0.2794, time=0.01s
[CLIPFilter debug] => wrote debug patch: goats2_patch19_a_goat’s_back_visible_in_tall_grass.jpg
[CLIPFilter] mask=20, best_label='sign', score=0.2679, time=0.01s
[CLIPFilter debug] => wrote debug patch: goats2_patch20_a_sign_stuck_in_the_ground_with_a_number.jpg
[CLIPFilter] mask=21, best_label='goat', score=0.3380, time=0.01s
[CLIPFilter debug] => wrote debug patch: goats2_patch21_a_brown_and_white_goat_in_the_middle_of_the_pasture.jpg
[CLIPFilter] mask=22, best_label='negative', score=0.2836, time=0.01s
[CLIPFilter debug] => wrote debug patch: goats2_patch22_flower_heads_on_thin_stems.jpg
[CLIPFilter] mask=23, best_label='negative', score=0.2689, time=0.01s
[CLIPFilter debug] => wrote debug patch: goats2_patch23_a_plastic_tie_or_string_in_grass.jpg
[CLIPFilter] mask=24, best_label='sign', score=0.2977, time=0.01s
[CLIPFilter debug] => wrote debug patch: goats2_patch24_a_labeled_field_marker_in_experimental_grass.jpg
[CLIPFilter] mask=25, best_label='sign', score=0.2954, time=0.01s
[CLIPFilter debug] => wrote debug patch: goats2_patch25_a_numbered_white_rectangle_in_tall_grass.jpg
[CLIPFilter] mask=26, best_label='sign', score=0.2861, time=0.01s
[CLIPFilter debug] => wrote debug patch: goats2_patch26_a_grassy_field_tag_with_black_digits.jpg
[CLIPFilter] mask=27, best_label='sign', score=0.3504, time=0.01s
[CLIPFilter debug] => wrote debug patch: goats2_patch27_a_pole-mounted_sign_with_a_test_plot_number.jpg
[CLIPFilter] mask=28, best_label='sign', score=0.3325, time=0.01s
[CLIPFilter debug] => wrote debug patch: goats2_patch28_a_labeled_field_marker_in_experimental_grass.jpg
[CLIPFilter] mask=29, best_label='sign', score=0.3367, time=0.01s
[CLIPFilter debug] => wrote debug patch: goats2_patch29_a_labeled_plot_sign_on_a_post.jpg
[CLIPFilter] mask=30, best_label='sign', score=0.3338, time=0.01s
[CLIPFilter debug] => wrote debug patch: goats2_patch30_a_pole-mounted_sign_with_a_test_plot_number.jpg
[CLIPFilter] mask=31, best_label='negative', score=0.2695, time=0.01s
[CLIPFilter debug] => wrote debug patch: goats2_patch31_a_blurry_object_between_wires.jpg
[CLIPFilter] mask=32, best_label='negative', score=0.2778, time=0.01s
[CLIPFilter debug] => wrote debug patch: goats2_patch32_a_dark_vertical_object_in_the_ground.jpg
[CLIPFilter] mask=33, best_label='sign', score=0.2932, time=0.01s
[CLIPFilter debug] => wrote debug patch: goats2_patch33_a_numbered_white_rectangle_in_tall_grass.jpg
[CLIPFilter] mask=34, best_label='sign', score=0.3395, time=0.01s
[CLIPFilter debug] => wrote debug patch: goats2_patch34_a_numbered_placard_staked_in_soil.jpg
[CLIPFilter] mask=35, best_label='negative', score=0.2833, time=0.01s
[CLIPFilter debug] => wrote debug patch: goats2_patch35_a_vertical_fence_stake_without_a_label.jpg
[CLIPFilter] mask=36, best_label='negative', score=0.2691, time=0.01s
[CLIPFilter debug] => wrote debug patch: goats2_patch36_a_group_of_overlapping_stems.jpg
[CLIPFilter] mask=37, best_label='negative', score=0.2895, time=0.01s
[CLIPFilter debug] => wrote debug patch: goats2_patch37_a_section_of_grass_with_white_dots.jpg
[CLIPFilter] mask=38, best_label='sign', score=0.3589, time=0.01s
[CLIPFilter debug] => wrote debug patch: goats2_patch38_a_pole-mounted_sign_with_a_test_plot_number.jpg
[CLIPFilter] mask=39, best_label='sign', score=0.3494, time=0.01s
[CLIPFilter debug] => wrote debug patch: goats2_patch39_a_number_marker_sign_above_vegetation.jpg
[CLIPFilter] mask=40, best_label='sign', score=0.3529, time=0.01s
[CLIPFilter debug] => wrote debug patch: goats2_patch40_a_printed_sign_showing_a_field_number.jpg
[CLIPFilter] mask=41, best_label='sign', score=0.3462, time=0.01s
[CLIPFilter debug] => wrote debug patch: goats2_patch41_a_square_white_field_marker_with_digits.jpg
[CLIPFilter] mask=42, best_label='sign', score=0.2880, time=0.01s
[CLIPFilter debug] => wrote debug patch: goats2_patch42_a_numbered_white_rectangle_in_tall_grass.jpg
[CLIPFilter] mask=43, best_label='sign', score=0.2754, time=0.01s
[CLIPFilter debug] => wrote debug patch: goats2_patch43_a_white_ID_sign_above_the_grass_line.jpg
[CLIPFilter] mask=44, best_label='sign', score=0.3208, time=0.01s
[CLIPFilter debug] => wrote debug patch: goats2_patch44_a_white_tag_showing_an_agricultural_zone_number.jpg
[CLIPFilter] mask=45, best_label='sign', score=0.3306, time=0.01s
[CLIPFilter debug] => wrote debug patch: goats2_patch45_a_square_white_field_marker_with_digits.jpg
[CLIPFilter] mask=46, best_label='sign', score=0.3506, time=0.01s
[CLIPFilter debug] => wrote debug patch: goats2_patch46_a_sign_stuck_in_the_ground_with_a_number.jpg
[CLIPFilter] mask=47, best_label='sign', score=0.3389, time=0.01s
[CLIPFilter debug] => wrote debug patch: goats2_patch47_a_pole-mounted_sign_with_a_test_plot_number.jpg
[CLIPFilter] mask=48, best_label='sign', score=0.3512, time=0.01s
[CLIPFilter debug] => wrote debug patch: goats2_patch48_a_square_white_field_marker_with_digits.jpg
[CLIPFilter] mask=49, best_label='sign', score=0.2857, time=0.01s
[CLIPFilter debug] => wrote debug patch: goats2_patch49_a_numbered_white_rectangle_in_tall_grass.jpg
[CLIPFilter] mask=50, best_label='sign', score=0.3396, time=0.01s
[CLIPFilter debug] => wrote debug patch: goats2_patch50_a_pole-mounted_sign_with_a_test_plot_number.jpg
[CLIPFilter] mask=51, best_label='sign', score=0.3446, time=0.01s
[CLIPFilter debug] => wrote debug patch: goats2_patch51_a_white_tag_showing_an_agricultural_zone_number.jpg
[CLIPFilter] mask=52, best_label='sign', score=0.3446, time=0.01s
[CLIPFilter debug] => wrote debug patch: goats2_patch52_a_white_tag_showing_an_agricultural_zone_number.jpg
[CLIPFilter] mask=53, best_label='sign', score=0.3041, time=0.01s
[CLIPFilter debug] => wrote debug patch: goats2_patch53_a_white_ID_sign_above_the_grass_line.jpg
[CLIPFilter] mask=54, best_label='sign', score=0.2896, time=0.01s
[CLIPFilter debug] => wrote debug patch: goats2_patch54_a_sign_stuck_in_the_ground_with_a_number.jpg
[CLIPFilter] mask=55, best_label='negative', score=0.2649, time=0.01s
[CLIPFilter debug] => wrote debug patch: goats2_patch55_a_shadow_line_in_the_vegetation.jpg
[CLIPFilter] mask=56, best_label='goat', score=0.3565, time=0.01s
[CLIPFilter debug] => wrote debug patch: goats2_patch56_a_brown_and_white_goat_in_the_middle_of_the_pasture.jpg
[CLIPFilter] mask=57, best_label='sign', score=0.3229, time=0.01s
[CLIPFilter debug] => wrote debug patch: goats2_patch57_a_white_ID_sign_above_the_grass_line.jpg
[CLIPFilter] mask=58, best_label='goat', score=0.2910, time=0.01s
[CLIPFilter debug] => wrote debug patch: goats2_patch58_a_goat’s_back_visible_in_tall_grass.jpg
[clip_filter] => classification done, now final label filter...
[postsam2processing debug] => saving final patches after classification...
  => wrote final patch => goats2_sam2-filtered-patch0000.jpg
  => wrote final patch => goats2_sam2-filtered-patch0001.jpg
  => wrote final patch => goats2_sam2-filtered-patch0002.jpg
  => wrote final patch => goats2_sam2-filtered-patch0003.jpg
  => wrote final patch => goats2_sam2-filtered-patch0004.jpg
  => wrote final patch => goats2_sam2-filtered-patch0005.jpg
  => wrote final patch => goats2_sam2-filtered-patch0006.jpg
  => wrote final patch => goats2_sam2-filtered-patch0007.jpg
  => wrote final patch => goats2_sam2-filtered-patch0008.jpg
[visualization] => building 'pre' 2x2 composite (sam2) ...
  => [build_composite_for_masks] building annotated overlay...
  => [build_composite_for_masks] building random color + masked array...
  => [build_composite_for_masks] building 2x2 now...
[visualization] => building 'post' 2x2 composite (clip) ...
  => [build_composite_for_masks] building annotated overlay...
  => [build_composite_for_masks] building random color + masked array...
  => [build_composite_for_masks] building 2x2 now...
[visualization] => generating panoptic final image with detectron2 Visualizer...
[visualization] => wrote final single overlay => demos/goats/output/goats2-final.jpg
[process_folder] => wrote JSON => demos/goats/output/goats2.json
[visualization] => wrote summary => demos/goats/output/goats2_summary.jpg
[process_folder] => done with image.

Done.

(sam2env) [jpers@wn212 zap-it]$ python zap-it-batch.py --config configs/tomato.yaml --dir demos/tomato/ --verbose full
Starting script...
[segment_images] Building SAM2 model...
[prepare_dirs] Removing old output: demos/tomato/output
[prepare_dirs] Created output folder: demos/tomato/output
[CLIPFilter] loading clip-vit-base-patch32
Using a slow image processor as `use_fast` is unset and a slow processor was saved with this model. `use_fast=True` will be the default behavior in v4.52, even if the model was saved with a slow processor. This will result in minor differences in outputs. You'll still be able to use a slow processor with `use_fast=False`.

[process_folder] => Handling image: 2022-07-22-16-25-44-48.jpg
 => Original shape = 1280x720
 => Single pass @native
[process_single_pass] Generating masks (single pass)...
[process_single_pass] => got 245 masks total.
[postsam2processing] => from 245 => 238 remain by area/box
[clip_filter] => classifying 238 bounding boxes...
[CLIPFilter] mask=0, best_label='negative', score=0.2746, time=0.01s
[CLIPFilter debug] => wrote debug patch: 2022-07-22-16-25-44-48_patch0_a_fluorescent_lamp.jpg
[CLIPFilter] mask=1, best_label='tomato_leaf', score=0.3022, time=0.01s
[CLIPFilter debug] => wrote debug patch: 2022-07-22-16-25-44-48_patch1_a_tomato_vine_leaf_pair.jpg
[CLIPFilter] mask=2, best_label='negative', score=0.2647, time=0.01s
[CLIPFilter debug] => wrote debug patch: 2022-07-22-16-25-44-48_patch2_a_binder_clip.jpg
[CLIPFilter] mask=3, best_label='green_tomato', score=0.3236, time=0.01s
[CLIPFilter debug] => wrote debug patch: 2022-07-22-16-25-44-48_patch3_a_green_tomato_under_foliage.jpg
[CLIPFilter] mask=4, best_label='negative', score=0.2545, time=0.01s
[CLIPFilter debug] => wrote debug patch: 2022-07-22-16-25-44-48_patch4_a_halogen_lamp.jpg
[CLIPFilter] mask=5, best_label='tomato_leaf', score=0.2580, time=0.01s
[CLIPFilter debug] => wrote debug patch: 2022-07-22-16-25-44-48_patch5_a_tomato_leaf_silhouette.jpg
[CLIPFilter] mask=6, best_label='green_tomato', score=0.3071, time=0.01s
[CLIPFilter debug] => wrote debug patch: 2022-07-22-16-25-44-48_patch6_a_green_tomato_hide_under_leaves.jpg
[CLIPFilter] mask=7, best_label='negative', score=0.2721, time=0.01s
[CLIPFilter debug] => wrote debug patch: 2022-07-22-16-25-44-48_patch7_a_fluorescent_lamp.jpg
[CLIPFilter] mask=8, best_label='tomato_leaf', score=0.2773, time=0.01s
[CLIPFilter debug] => wrote debug patch: 2022-07-22-16-25-44-48_patch8_a_tomato_leaf_midrib_visible.jpg
[CLIPFilter] mask=9, best_label='tomato_leaf', score=0.2811, time=0.01s
[CLIPFilter debug] => wrote debug patch: 2022-07-22-16-25-44-48_patch9_a_tomato_leaf_midrib_visible.jpg
[CLIPFilter] mask=10, best_label='negative', score=0.2597, time=0.01s
[CLIPFilter debug] => wrote debug patch: 2022-07-22-16-25-44-48_patch10_a_metal_nut.jpg
[CLIPFilter] mask=11, best_label='negative', score=0.2510, time=0.01s
[CLIPFilter debug] => wrote debug patch: 2022-07-22-16-25-44-48_patch11_a_drip_line.jpg
[CLIPFilter] mask=12, best_label='tomato_leaf', score=0.2959, time=0.01s
[CLIPFilter debug] => wrote debug patch: 2022-07-22-16-25-44-48_patch12_a_tomato_leaf_midrib_visible.jpg
[CLIPFilter] mask=13, best_label='tomato_leaf', score=0.2718, time=0.01s
[CLIPFilter debug] => wrote debug patch: 2022-07-22-16-25-44-48_patch13_a_tomato_leaf_silhouette.jpg
[CLIPFilter] mask=14, best_label='negative', score=0.2765, time=0.01s
[CLIPFilter debug] => wrote debug patch: 2022-07-22-16-25-44-48_patch14_a_thermal_screen.jpg
[CLIPFilter] mask=15, best_label='green_tomato', score=0.2840, time=0.01s
[CLIPFilter debug] => wrote debug patch: 2022-07-22-16-25-44-48_patch15_a_green_tomato_hide_under_leaves.jpg
[CLIPFilter] mask=16, best_label='green_tomato', score=0.2932, time=0.01s
[CLIPFilter debug] => wrote debug patch: 2022-07-22-16-25-44-48_patch16_a_green_tomato_in_lower_truss.jpg
[CLIPFilter] mask=17, best_label='negative', score=0.2709, time=0.01s
[CLIPFilter debug] => wrote debug patch: 2022-07-22-16-25-44-48_patch17_a_fluorescent_lamp.jpg
[CLIPFilter] mask=18, best_label='negative', score=0.2596, time=0.01s
[CLIPFilter debug] => wrote debug patch: 2022-07-22-16-25-44-48_patch18_a_black_hose.jpg
[CLIPFilter] mask=19, best_label='negative', score=0.2459, time=0.01s
[CLIPFilter debug] => wrote debug patch: 2022-07-22-16-25-44-48_patch19_a_support_post.jpg
[CLIPFilter] mask=20, best_label='tomato_leaf', score=0.3128, time=0.01s
[CLIPFilter debug] => wrote debug patch: 2022-07-22-16-25-44-48_patch20_a_tomato_leaf_in_full_shade.jpg
[CLIPFilter] mask=21, best_label='negative', score=0.2712, time=0.01s
[CLIPFilter debug] => wrote debug patch: 2022-07-22-16-25-44-48_patch21_a_fluorescent_lamp.jpg
[CLIPFilter] mask=22, best_label='tomato_leaf', score=0.3141, time=0.01s
[CLIPFilter debug] => wrote debug patch: 2022-07-22-16-25-44-48_patch22_a_tomato_leaf_in_full_shade.jpg
[CLIPFilter] mask=23, best_label='green_tomato', score=0.2864, time=0.01s
[CLIPFilter debug] => wrote debug patch: 2022-07-22-16-25-44-48_patch23_a_green_tomato_hide_under_leaves.jpg
[CLIPFilter] mask=24, best_label='negative', score=0.2858, time=0.01s
[CLIPFilter debug] => wrote debug patch: 2022-07-22-16-25-44-48_patch24_a_plastic_gutter.jpg
[CLIPFilter] mask=25, best_label='green_tomato', score=0.2660, time=0.01s
[CLIPFilter debug] => wrote debug patch: 2022-07-22-16-25-44-48_patch25_a_tomato_with_dark_green_skin.jpg
[CLIPFilter] mask=26, best_label='negative', score=0.2741, time=0.01s
[CLIPFilter debug] => wrote debug patch: 2022-07-22-16-25-44-48_patch26_a_metal_hook.jpg
[CLIPFilter] mask=27, best_label='negative', score=0.2704, time=0.01s
[CLIPFilter debug] => wrote debug patch: 2022-07-22-16-25-44-48_patch27_a_plastic_tubing.jpg
[CLIPFilter] mask=28, best_label='green_tomato', score=0.2588, time=0.01s
[CLIPFilter debug] => wrote debug patch: 2022-07-22-16-25-44-48_patch28_a_tomato_with_dark_green_skin.jpg
[CLIPFilter] mask=29, best_label='green_tomato', score=0.2697, time=0.01s
[CLIPFilter debug] => wrote debug patch: 2022-07-22-16-25-44-48_patch29_a_green_tomato_in_lower_truss.jpg
[CLIPFilter] mask=30, best_label='negative', score=0.2640, time=0.01s
[CLIPFilter debug] => wrote debug patch: 2022-07-22-16-25-44-48_patch30_a_fluorescent_lamp.jpg
[CLIPFilter] mask=31, best_label='negative', score=0.2689, time=0.01s
[CLIPFilter debug] => wrote debug patch: 2022-07-22-16-25-44-48_patch31_a_plastic_gutter.jpg
[CLIPFilter] mask=32, best_label='negative', score=0.2677, time=0.01s
[CLIPFilter debug] => wrote debug patch: 2022-07-22-16-25-44-48_patch32_a_plastic_tubing.jpg
[CLIPFilter] mask=33, best_label='negative', score=0.2768, time=0.01s
[CLIPFilter debug] => wrote debug patch: 2022-07-22-16-25-44-48_patch33_a_plastic_tubing.jpg
[CLIPFilter] mask=34, best_label='negative', score=0.2718, time=0.01s
[CLIPFilter debug] => wrote debug patch: 2022-07-22-16-25-44-48_patch34_a_stake_tie.jpg
[CLIPFilter] mask=35, best_label='negative', score=0.2600, time=0.01s
[CLIPFilter debug] => wrote debug patch: 2022-07-22-16-25-44-48_patch35_a_fan_blade.jpg
[CLIPFilter] mask=36, best_label='tomato_leaf', score=0.2616, time=0.01s
[CLIPFilter debug] => wrote debug patch: 2022-07-22-16-25-44-48_patch36_a_tomato_leaf_silhouette.jpg
[CLIPFilter] mask=37, best_label='negative', score=0.2932, time=0.01s
[CLIPFilter debug] => wrote debug patch: 2022-07-22-16-25-44-48_patch37_a_pruning_shear.jpg
[CLIPFilter] mask=38, best_label='negative', score=0.2698, time=0.01s
[CLIPFilter debug] => wrote debug patch: 2022-07-22-16-25-44-48_patch38_an_identification_tag.jpg
[CLIPFilter] mask=39, best_label='green_tomato', score=0.2434, time=0.01s
[CLIPFilter debug] => wrote debug patch: 2022-07-22-16-25-44-48_patch39_a_small_green_tomato_orb.jpg
[CLIPFilter] mask=40, best_label='negative', score=0.2677, time=0.01s
[CLIPFilter debug] => wrote debug patch: 2022-07-22-16-25-44-48_patch40_a_plastic_tubing.jpg
[CLIPFilter] mask=41, best_label='negative', score=0.3009, time=0.01s
[CLIPFilter debug] => wrote debug patch: 2022-07-22-16-25-44-48_patch41_an_electrical_cable.jpg
[CLIPFilter] mask=42, best_label='negative', score=0.2783, time=0.01s
[CLIPFilter debug] => wrote debug patch: 2022-07-22-16-25-44-48_patch42_a_greenhouse_crossbeam.jpg
[CLIPFilter] mask=43, best_label='negative', score=0.3004, time=0.01s
[CLIPFilter debug] => wrote debug patch: 2022-07-22-16-25-44-48_patch43_a_metal_stake.jpg
[CLIPFilter] mask=44, best_label='negative', score=0.2592, time=0.01s
[CLIPFilter debug] => wrote debug patch: 2022-07-22-16-25-44-48_patch44_a_support_wire.jpg
[CLIPFilter] mask=45, best_label='negative', score=0.2517, time=0.01s
[CLIPFilter debug] => wrote debug patch: 2022-07-22-16-25-44-48_patch45_a_plastic_tray.jpg
[CLIPFilter] mask=46, best_label='negative', score=0.2761, time=0.01s
[CLIPFilter debug] => wrote debug patch: 2022-07-22-16-25-44-48_patch46_a_zip_tie.jpg
[CLIPFilter] mask=47, best_label='negative', score=0.2755, time=0.01s
[CLIPFilter debug] => wrote debug patch: 2022-07-22-16-25-44-48_patch47_a_black_hose.jpg
[CLIPFilter] mask=48, best_label='negative', score=0.2552, time=0.01s
[CLIPFilter debug] => wrote debug patch: 2022-07-22-16-25-44-48_patch48_a_hydroponic_channel.jpg
[CLIPFilter] mask=49, best_label='negative', score=0.2977, time=0.01s
[CLIPFilter debug] => wrote debug patch: 2022-07-22-16-25-44-48_patch49_an_electrical_cable.jpg
[CLIPFilter] mask=50, best_label='negative', score=0.2740, time=0.01s
[CLIPFilter debug] => wrote debug patch: 2022-07-22-16-25-44-48_patch50_a_fluorescent_lamp.jpg
[CLIPFilter] mask=51, best_label='negative', score=0.3052, time=0.01s
[CLIPFilter debug] => wrote debug patch: 2022-07-22-16-25-44-48_patch51_a_plastic_tubing.jpg
[CLIPFilter] mask=52, best_label='negative', score=0.2680, time=0.01s
[CLIPFilter debug] => wrote debug patch: 2022-07-22-16-25-44-48_patch52_a_stake_tie.jpg
[CLIPFilter] mask=53, best_label='negative', score=0.2778, time=0.01s
[CLIPFilter debug] => wrote debug patch: 2022-07-22-16-25-44-48_patch53_a_zip_tie.jpg
[CLIPFilter] mask=54, best_label='negative', score=0.2625, time=0.01s
[CLIPFilter debug] => wrote debug patch: 2022-07-22-16-25-44-48_patch54_a_thermal_screen.jpg
[CLIPFilter] mask=55, best_label='negative', score=0.2596, time=0.01s
[CLIPFilter debug] => wrote debug patch: 2022-07-22-16-25-44-48_patch55_a_stake_tie.jpg
[CLIPFilter] mask=56, best_label='negative', score=0.2740, time=0.01s
[CLIPFilter debug] => wrote debug patch: 2022-07-22-16-25-44-48_patch56_a_greenhouse_crossbeam.jpg
[CLIPFilter] mask=57, best_label='negative', score=0.2789, time=0.01s
[CLIPFilter debug] => wrote debug patch: 2022-07-22-16-25-44-48_patch57_a_greenhouse_crossbeam.jpg
[CLIPFilter] mask=58, best_label='negative', score=0.2727, time=0.01s
[CLIPFilter debug] => wrote debug patch: 2022-07-22-16-25-44-48_patch58_a_zip_tie.jpg
[CLIPFilter] mask=59, best_label='negative', score=0.2746, time=0.01s
[CLIPFilter debug] => wrote debug patch: 2022-07-22-16-25-44-48_patch59_a_zip_tie.jpg
[CLIPFilter] mask=60, best_label='ripe_tomato', score=0.3458, time=0.01s
[CLIPFilter debug] => wrote debug patch: 2022-07-22-16-25-44-48_patch60_a_well-ripened_tomato_on_branch.jpg
[CLIPFilter] mask=61, best_label='negative', score=0.2648, time=0.01s
[CLIPFilter debug] => wrote debug patch: 2022-07-22-16-25-44-48_patch61_a_reflective_foil.jpg
[CLIPFilter] mask=62, best_label='ripe_tomato', score=0.3192, time=0.01s
[CLIPFilter debug] => wrote debug patch: 2022-07-22-16-25-44-48_patch62_a_well-ripened_tomato_on_branch.jpg
[CLIPFilter] mask=63, best_label='tomato_leaf', score=0.2648, time=0.01s
[CLIPFilter debug] => wrote debug patch: 2022-07-22-16-25-44-48_patch63_a_tomato_leaf_in_greenhouse_lighting.jpg
[CLIPFilter] mask=64, best_label='negative', score=0.2422, time=0.01s
[CLIPFilter debug] => wrote debug patch: 2022-07-22-16-25-44-48_patch64_a_binder_clip.jpg
[CLIPFilter] mask=65, best_label='green_tomato', score=0.2606, time=0.01s
[CLIPFilter debug] => wrote debug patch: 2022-07-22-16-25-44-48_patch65_a_tomato_with_dark_green_skin.jpg
[CLIPFilter] mask=66, best_label='negative', score=0.2611, time=0.01s
[CLIPFilter debug] => wrote debug patch: 2022-07-22-16-25-44-48_patch66_a_thermal_screen.jpg
[CLIPFilter] mask=67, best_label='green_tomato', score=0.2692, time=0.01s
[CLIPFilter debug] => wrote debug patch: 2022-07-22-16-25-44-48_patch67_a_cluster_of_firm_green_spheres.jpg
[CLIPFilter] mask=68, best_label='tomato_leaf', score=0.2922, time=0.01s
[CLIPFilter debug] => wrote debug patch: 2022-07-22-16-25-44-48_patch68_a_tomato_leaf_in_greenhouse_lighting.jpg
[CLIPFilter] mask=69, best_label='negative', score=0.2693, time=0.01s
[CLIPFilter debug] => wrote debug patch: 2022-07-22-16-25-44-48_patch69_a_reflective_foil.jpg
[CLIPFilter] mask=70, best_label='negative', score=0.2808, time=0.01s
[CLIPFilter debug] => wrote debug patch: 2022-07-22-16-25-44-48_patch70_a_tomato_vine_segment.jpg
[CLIPFilter] mask=71, best_label='negative', score=0.2733, time=0.01s
[CLIPFilter debug] => wrote debug patch: 2022-07-22-16-25-44-48_patch71_a_shading_screen.jpg
[CLIPFilter] mask=72, best_label='negative', score=0.2527, time=0.01s
[CLIPFilter debug] => wrote debug patch: 2022-07-22-16-25-44-48_patch72_a_fluorescent_lamp.jpg
[CLIPFilter] mask=73, best_label='negative', score=0.2837, time=0.01s
[CLIPFilter debug] => wrote debug patch: 2022-07-22-16-25-44-48_patch73_a_fluorescent_lamp.jpg
[CLIPFilter] mask=74, best_label='negative', score=0.2920, time=0.01s
[CLIPFilter debug] => wrote debug patch: 2022-07-22-16-25-44-48_patch74_a_greenhouse_crossbeam.jpg
[CLIPFilter] mask=75, best_label='negative', score=0.2428, time=0.01s
[CLIPFilter debug] => wrote debug patch: 2022-07-22-16-25-44-48_patch75_a_fluorescent_lamp.jpg
[CLIPFilter] mask=76, best_label='negative', score=0.2644, time=0.01s
[CLIPFilter debug] => wrote debug patch: 2022-07-22-16-25-44-48_patch76_a_fluorescent_lamp.jpg
[CLIPFilter] mask=77, best_label='negative', score=0.2728, time=0.01s
[CLIPFilter debug] => wrote debug patch: 2022-07-22-16-25-44-48_patch77_a_fluorescent_lamp.jpg
[CLIPFilter] mask=78, best_label='ripe_tomato', score=0.2842, time=0.01s
[CLIPFilter debug] => wrote debug patch: 2022-07-22-16-25-44-48_patch78_a_red_tomato_on_a_green_vine.jpg
[CLIPFilter] mask=79, best_label='negative', score=0.2780, time=0.01s
[CLIPFilter debug] => wrote debug patch: 2022-07-22-16-25-44-48_patch79_a_fluorescent_lamp.jpg
[CLIPFilter] mask=80, best_label='ripe_tomato', score=0.3348, time=0.01s
[CLIPFilter debug] => wrote debug patch: 2022-07-22-16-25-44-48_patch80_a_red_tomato_cluster_in_greenhouse.jpg
[CLIPFilter] mask=81, best_label='ripe_tomato', score=0.3268, time=0.01s
[CLIPFilter debug] => wrote debug patch: 2022-07-22-16-25-44-48_patch81_a_crimson_red_ripe_tomato.jpg
[CLIPFilter] mask=82, best_label='ripe_tomato', score=0.3393, time=0.01s
[CLIPFilter debug] => wrote debug patch: 2022-07-22-16-25-44-48_patch82_a_mature_red_tomato.jpg
[CLIPFilter] mask=83, best_label='green_tomato', score=0.2934, time=0.01s
[CLIPFilter debug] => wrote debug patch: 2022-07-22-16-25-44-48_patch83_a_green_tomato_hanging.jpg
[CLIPFilter] mask=84, best_label='ripe_tomato', score=0.3013, time=0.01s
[CLIPFilter debug] => wrote debug patch: 2022-07-22-16-25-44-48_patch84_a_red_tomato_on_a_green_vine.jpg
[CLIPFilter] mask=85, best_label='negative', score=0.2807, time=0.01s
[CLIPFilter debug] => wrote debug patch: 2022-07-22-16-25-44-48_patch85_a_fan_blade.jpg
[CLIPFilter] mask=86, best_label='negative', score=0.2694, time=0.01s
[CLIPFilter debug] => wrote debug patch: 2022-07-22-16-25-44-48_patch86_a_greenhouse_sign.jpg
[CLIPFilter] mask=87, best_label='ripe_tomato', score=0.3129, time=0.01s
[CLIPFilter debug] => wrote debug patch: 2022-07-22-16-25-44-48_patch87_a_red_tomato_on_a_green_vine.jpg
[CLIPFilter] mask=88, best_label='negative', score=0.2556, time=0.01s
[CLIPFilter debug] => wrote debug patch: 2022-07-22-16-25-44-48_patch88_a_greenhouse_crossbeam.jpg
[CLIPFilter] mask=89, best_label='negative', score=0.2746, time=0.01s
[CLIPFilter debug] => wrote debug patch: 2022-07-22-16-25-44-48_patch89_a_greenhouse_sign.jpg
[CLIPFilter] mask=90, best_label='green_tomato', score=0.2821, time=0.01s
[CLIPFilter debug] => wrote debug patch: 2022-07-22-16-25-44-48_patch90_a_green_tomato_in_lower_truss.jpg
[CLIPFilter] mask=91, best_label='negative', score=0.2765, time=0.01s
[CLIPFilter debug] => wrote debug patch: 2022-07-22-16-25-44-48_patch91_a_metal_nut.jpg
[CLIPFilter] mask=92, best_label='green_tomato', score=0.2783, time=0.01s
[CLIPFilter debug] => wrote debug patch: 2022-07-22-16-25-44-48_patch92_a_small_green_tomato_orb.jpg
[CLIPFilter] mask=93, best_label='ripe_tomato', score=0.3406, time=0.01s
[CLIPFilter debug] => wrote debug patch: 2022-07-22-16-25-44-48_patch93_a_red_tomato_hanging_on_stem.jpg
[CLIPFilter] mask=94, best_label='ripe_tomato', score=0.3041, time=0.01s
[CLIPFilter debug] => wrote debug patch: 2022-07-22-16-25-44-48_patch94_a_mature_red_tomato.jpg
[CLIPFilter] mask=95, best_label='ripe_tomato', score=0.3343, time=0.01s
[CLIPFilter debug] => wrote debug patch: 2022-07-22-16-25-44-48_patch95_a_round_red_ripe_tomato.jpg
[CLIPFilter] mask=96, best_label='green_tomato', score=0.2732, time=0.01s
[CLIPFilter debug] => wrote debug patch: 2022-07-22-16-25-44-48_patch96_a_green_tomato_in_greenhouse.jpg
[CLIPFilter] mask=97, best_label='negative', score=0.2354, time=0.01s
[CLIPFilter debug] => wrote debug patch: 2022-07-22-16-25-44-48_patch97_an_S-hook.jpg
[CLIPFilter] mask=98, best_label='half_ripe_tomato', score=0.3078, time=0.01s
[CLIPFilter debug] => wrote debug patch: 2022-07-22-16-25-44-48_patch98_a_tomato_with_red_shoulders.jpg
[CLIPFilter] mask=99, best_label='negative', score=0.2727, time=0.01s
[CLIPFilter debug] => wrote debug patch: 2022-07-22-16-25-44-48_patch99_a_stake_tie.jpg
[CLIPFilter] mask=100, best_label='negative', score=0.2518, time=0.01s
[CLIPFilter debug] => wrote debug patch: 2022-07-22-16-25-44-48_patch100_a_white_board.jpg
[CLIPFilter] mask=101, best_label='negative', score=0.2910, time=0.01s
[CLIPFilter debug] => wrote debug patch: 2022-07-22-16-25-44-48_patch101_a_greenhouse_crossbeam.jpg
[CLIPFilter] mask=102, best_label='negative', score=0.2594, time=0.01s
[CLIPFilter debug] => wrote debug patch: 2022-07-22-16-25-44-48_patch102_an_identification_tag.jpg
[CLIPFilter] mask=103, best_label='ripe_tomato', score=0.3017, time=0.01s
[CLIPFilter debug] => wrote debug patch: 2022-07-22-16-25-44-48_patch103_a_red_tomato_on_a_green_vine.jpg
[CLIPFilter] mask=104, best_label='negative', score=0.2969, time=0.01s
[CLIPFilter debug] => wrote debug patch: 2022-07-22-16-25-44-48_patch104_a_zip_tie.jpg
[CLIPFilter] mask=105, best_label='ripe_tomato', score=0.2869, time=0.01s
[CLIPFilter debug] => wrote debug patch: 2022-07-22-16-25-44-48_patch105_a_red_tomato_with_smooth_skin.jpg
[CLIPFilter] mask=106, best_label='negative', score=0.2905, time=0.01s
[CLIPFilter debug] => wrote debug patch: 2022-07-22-16-25-44-48_patch106_a_zip_tie.jpg
[CLIPFilter] mask=107, best_label='negative', score=0.3082, time=0.01s
[CLIPFilter debug] => wrote debug patch: 2022-07-22-16-25-44-48_patch107_an_electrical_cable.jpg
[CLIPFilter] mask=108, best_label='negative', score=0.2722, time=0.01s
[CLIPFilter debug] => wrote debug patch: 2022-07-22-16-25-44-48_patch108_a_greenhouse_crossbeam.jpg
[CLIPFilter] mask=109, best_label='negative', score=0.2835, time=0.01s
[CLIPFilter debug] => wrote debug patch: 2022-07-22-16-25-44-48_patch109_a_plastic_tubing.jpg
[CLIPFilter] mask=110, best_label='negative', score=0.2569, time=0.01s
[CLIPFilter debug] => wrote debug patch: 2022-07-22-16-25-44-48_patch110_a_roof_panel.jpg
[CLIPFilter] mask=111, best_label='negative', score=0.2989, time=0.01s
[CLIPFilter debug] => wrote debug patch: 2022-07-22-16-25-44-48_patch111_an_electrical_cable.jpg
[CLIPFilter] mask=112, best_label='tomato_leaf', score=0.3056, time=0.01s
[CLIPFilter debug] => wrote debug patch: 2022-07-22-16-25-44-48_patch112_a_tomato_leaf_in_greenhouse_lighting.jpg
[CLIPFilter] mask=113, best_label='half_ripe_tomato', score=0.3680, time=0.01s
[CLIPFilter debug] => wrote debug patch: 2022-07-22-16-25-44-48_patch113_a_tomato_showing_two-tone_coloring.jpg
[CLIPFilter] mask=114, best_label='half_ripe_tomato', score=0.2815, time=0.01s
[CLIPFilter debug] => wrote debug patch: 2022-07-22-16-25-44-48_patch114_a_tomato_with_red_shoulders.jpg
[CLIPFilter] mask=115, best_label='ripe_tomato', score=0.3091, time=0.01s
[CLIPFilter debug] => wrote debug patch: 2022-07-22-16-25-44-48_patch115_a_mature_red_tomato.jpg
[CLIPFilter] mask=116, best_label='ripe_tomato', score=0.3506, time=0.01s
[CLIPFilter debug] => wrote debug patch: 2022-07-22-16-25-44-48_patch116_a_plump_red_tomato_on_plant.jpg
[CLIPFilter] mask=117, best_label='ripe_tomato', score=0.3228, time=0.01s
[CLIPFilter debug] => wrote debug patch: 2022-07-22-16-25-44-48_patch117_a_glossy_red_tomato.jpg
[CLIPFilter] mask=118, best_label='half_ripe_tomato', score=0.2654, time=0.01s
[CLIPFilter debug] => wrote debug patch: 2022-07-22-16-25-44-48_patch118_a_tomato_between_green_and_red.jpg
[CLIPFilter] mask=119, best_label='negative', score=0.2789, time=0.01s
[CLIPFilter debug] => wrote debug patch: 2022-07-22-16-25-44-48_patch119_a_fluorescent_lamp.jpg
[CLIPFilter] mask=120, best_label='green_tomato', score=0.2394, time=0.01s
[CLIPFilter debug] => wrote debug patch: 2022-07-22-16-25-44-48_patch120_a_tomato_with_dark_green_skin.jpg
[CLIPFilter] mask=121, best_label='green_tomato', score=0.2563, time=0.01s
[CLIPFilter debug] => wrote debug patch: 2022-07-22-16-25-44-48_patch121_a_green_tomato_hide_under_leaves.jpg
[CLIPFilter] mask=122, best_label='negative', score=0.2578, time=0.01s
[CLIPFilter debug] => wrote debug patch: 2022-07-22-16-25-44-48_patch122_a_fan_blade.jpg
[CLIPFilter] mask=123, best_label='green_tomato', score=0.2804, time=0.01s
[CLIPFilter debug] => wrote debug patch: 2022-07-22-16-25-44-48_patch123_a_green_tomato_within_foliage_mass.jpg
[CLIPFilter] mask=124, best_label='negative', score=0.2937, time=0.01s
[CLIPFilter debug] => wrote debug patch: 2022-07-22-16-25-44-48_patch124_a_fluorescent_lamp.jpg
[CLIPFilter] mask=125, best_label='negative', score=0.2675, time=0.01s
[CLIPFilter debug] => wrote debug patch: 2022-07-22-16-25-44-48_patch125_a_fluorescent_lamp.jpg
[CLIPFilter] mask=126, best_label='negative', score=0.2819, time=0.01s
[CLIPFilter debug] => wrote debug patch: 2022-07-22-16-25-44-48_patch126_a_fan_blade.jpg
[CLIPFilter] mask=127, best_label='green_tomato', score=0.2846, time=0.01s
[CLIPFilter debug] => wrote debug patch: 2022-07-22-16-25-44-48_patch127_a_green_tomato_hide_under_leaves.jpg
[CLIPFilter] mask=128, best_label='negative', score=0.2833, time=0.01s
[CLIPFilter debug] => wrote debug patch: 2022-07-22-16-25-44-48_patch128_a_fluorescent_lamp.jpg
[CLIPFilter] mask=129, best_label='negative', score=0.2552, time=0.01s
[CLIPFilter debug] => wrote debug patch: 2022-07-22-16-25-44-48_patch129_a_fluorescent_lamp.jpg
[CLIPFilter] mask=130, best_label='negative', score=0.2856, time=0.01s
[CLIPFilter debug] => wrote debug patch: 2022-07-22-16-25-44-48_patch130_a_greenhouse_crossbeam.jpg
[CLIPFilter] mask=131, best_label='green_tomato', score=0.2460, time=0.01s
[CLIPFilter debug] => wrote debug patch: 2022-07-22-16-25-44-48_patch131_a_tomato_with_dark_green_skin.jpg
[CLIPFilter] mask=132, best_label='tomato_leaf', score=0.2869, time=0.01s
[CLIPFilter debug] => wrote debug patch: 2022-07-22-16-25-44-48_patch132_a_tomato_leaf_in_greenhouse_lighting.jpg
[CLIPFilter] mask=133, best_label='negative', score=0.2486, time=0.01s
[CLIPFilter debug] => wrote debug patch: 2022-07-22-16-25-44-48_patch133_a_thermal_screen.jpg
[CLIPFilter] mask=134, best_label='ripe_tomato', score=0.3549, time=0.01s
[CLIPFilter debug] => wrote debug patch: 2022-07-22-16-25-44-48_patch134_a_plump_red_tomato_on_plant.jpg
[CLIPFilter] mask=135, best_label='tomato_leaf', score=0.2699, time=0.01s
[CLIPFilter debug] => wrote debug patch: 2022-07-22-16-25-44-48_patch135_a_tomato_plant_foliage_section.jpg
[CLIPFilter] mask=136, best_label='negative', score=0.2689, time=0.01s
[CLIPFilter debug] => wrote debug patch: 2022-07-22-16-25-44-48_patch136_a_fan_blade.jpg
[CLIPFilter] mask=137, best_label='green_tomato', score=0.2699, time=0.01s
[CLIPFilter debug] => wrote debug patch: 2022-07-22-16-25-44-48_patch137_a_green_tomato_grouping_with_others.jpg
[CLIPFilter] mask=138, best_label='negative', score=0.2526, time=0.01s
[CLIPFilter debug] => wrote debug patch: 2022-07-22-16-25-44-48_patch138_a_thermal_screen.jpg
[CLIPFilter] mask=139, best_label='green_tomato', score=0.3035, time=0.01s
[CLIPFilter debug] => wrote debug patch: 2022-07-22-16-25-44-48_patch139_an_unripe_tomato_on_branch.jpg
[CLIPFilter] mask=140, best_label='green_tomato', score=0.2784, time=0.01s
[CLIPFilter debug] => wrote debug patch: 2022-07-22-16-25-44-48_patch140_a_green_tomato_in_greenhouse_row.jpg
[CLIPFilter] mask=141, best_label='green_tomato', score=0.2723, time=0.01s
[CLIPFilter debug] => wrote debug patch: 2022-07-22-16-25-44-48_patch141_a_green_tomato_grouping_with_others.jpg
[CLIPFilter] mask=142, best_label='ripe_tomato', score=0.3140, time=0.01s
[CLIPFilter debug] => wrote debug patch: 2022-07-22-16-25-44-48_patch142_a_red_tomato_fruit_on_tomato_plant.jpg
[CLIPFilter] mask=143, best_label='negative', score=0.2518, time=0.01s
[CLIPFilter debug] => wrote debug patch: 2022-07-22-16-25-44-48_patch143_a_thermal_screen.jpg
[CLIPFilter] mask=144, best_label='ripe_tomato', score=0.3010, time=0.01s
[CLIPFilter debug] => wrote debug patch: 2022-07-22-16-25-44-48_patch144_a_red_tomato_on_a_green_vine.jpg
[CLIPFilter] mask=145, best_label='ripe_tomato', score=0.2889, time=0.01s
[CLIPFilter debug] => wrote debug patch: 2022-07-22-16-25-44-48_patch145_a_red_tomato_on_a_green_vine.jpg
[CLIPFilter] mask=146, best_label='negative', score=0.2654, time=0.01s
[CLIPFilter debug] => wrote debug patch: 2022-07-22-16-25-44-48_patch146_a_metal_frame.jpg
[CLIPFilter] mask=147, best_label='negative', score=0.2544, time=0.01s
[CLIPFilter debug] => wrote debug patch: 2022-07-22-16-25-44-48_patch147_a_signage_board.jpg
[CLIPFilter] mask=148, best_label='green_tomato', score=0.2680, time=0.01s
[CLIPFilter debug] => wrote debug patch: 2022-07-22-16-25-44-48_patch148_a_green_tomato_hanging.jpg
[CLIPFilter] mask=149, best_label='ripe_tomato', score=0.3330, time=0.01s
[CLIPFilter debug] => wrote debug patch: 2022-07-22-16-25-44-48_patch149_a_red_tomato_cluster_in_greenhouse.jpg
[CLIPFilter] mask=150, best_label='green_tomato', score=0.2642, time=0.01s
[CLIPFilter debug] => wrote debug patch: 2022-07-22-16-25-44-48_patch150_a_tomato_with_dark_green_skin.jpg
[CLIPFilter] mask=151, best_label='negative', score=0.2679, time=0.01s
[CLIPFilter debug] => wrote debug patch: 2022-07-22-16-25-44-48_patch151_a_watering_nozzle.jpg
[CLIPFilter] mask=152, best_label='negative', score=0.2641, time=0.01s
[CLIPFilter debug] => wrote debug patch: 2022-07-22-16-25-44-48_patch152_a_signage_board.jpg
[CLIPFilter] mask=153, best_label='ripe_tomato', score=0.2989, time=0.01s
[CLIPFilter debug] => wrote debug patch: 2022-07-22-16-25-44-48_patch153_a_red_tomato_cluster_in_greenhouse.jpg
[CLIPFilter] mask=154, best_label='negative', score=0.2928, time=0.01s
[CLIPFilter debug] => wrote debug patch: 2022-07-22-16-25-44-48_patch154_an_electrical_cable.jpg
[CLIPFilter] mask=155, best_label='half_ripe_tomato', score=0.2553, time=0.01s
[CLIPFilter debug] => wrote debug patch: 2022-07-22-16-25-44-48_patch155_a_tomato_with_red_shoulders.jpg
[CLIPFilter] mask=156, best_label='negative', score=0.2537, time=0.01s
[CLIPFilter debug] => wrote debug patch: 2022-07-22-16-25-44-48_patch156_a_white_board.jpg
[CLIPFilter] mask=157, best_label='negative', score=0.2459, time=0.01s
[CLIPFilter debug] => wrote debug patch: 2022-07-22-16-25-44-48_patch157_a_white_board.jpg
[CLIPFilter] mask=158, best_label='ripe_tomato', score=0.3300, time=0.01s
[CLIPFilter debug] => wrote debug patch: 2022-07-22-16-25-44-48_patch158_a_well-ripened_tomato_on_branch.jpg
[CLIPFilter] mask=159, best_label='green_tomato', score=0.2764, time=0.01s
[CLIPFilter debug] => wrote debug patch: 2022-07-22-16-25-44-48_patch159_a_green_tomato_hide_under_leaves.jpg
[CLIPFilter] mask=160, best_label='negative', score=0.2997, time=0.01s
[CLIPFilter debug] => wrote debug patch: 2022-07-22-16-25-44-48_patch160_a_zip_tie.jpg
[CLIPFilter] mask=161, best_label='negative', score=0.2506, time=0.01s
[CLIPFilter debug] => wrote debug patch: 2022-07-22-16-25-44-48_patch161_a_support_post.jpg
[CLIPFilter] mask=162, best_label='negative', score=0.2679, time=0.01s
[CLIPFilter debug] => wrote debug patch: 2022-07-22-16-25-44-48_patch162_a_greenhouse_crossbeam.jpg
[CLIPFilter] mask=163, best_label='negative', score=0.2529, time=0.01s
[CLIPFilter debug] => wrote debug patch: 2022-07-22-16-25-44-48_patch163_a_white_board.jpg
[CLIPFilter] mask=164, best_label='ripe_tomato', score=0.3040, time=0.01s
[CLIPFilter debug] => wrote debug patch: 2022-07-22-16-25-44-48_patch164_a_red_tomato_on_a_green_vine.jpg
[CLIPFilter] mask=165, best_label='ripe_tomato', score=0.3020, time=0.01s
[CLIPFilter debug] => wrote debug patch: 2022-07-22-16-25-44-48_patch165_a_crimson_red_ripe_tomato.jpg
[CLIPFilter] mask=166, best_label='ripe_tomato', score=0.3196, time=0.01s
[CLIPFilter debug] => wrote debug patch: 2022-07-22-16-25-44-48_patch166_a_crimson_red_ripe_tomato.jpg
[CLIPFilter] mask=167, best_label='ripe_tomato', score=0.2939, time=0.01s
[CLIPFilter debug] => wrote debug patch: 2022-07-22-16-25-44-48_patch167_a_crimson_red_ripe_tomato.jpg
[CLIPFilter] mask=168, best_label='ripe_tomato', score=0.2760, time=0.01s
[CLIPFilter debug] => wrote debug patch: 2022-07-22-16-25-44-48_patch168_a_glossy_red_tomato.jpg
[CLIPFilter] mask=169, best_label='negative', score=0.3055, time=0.01s
[CLIPFilter debug] => wrote debug patch: 2022-07-22-16-25-44-48_patch169_a_halogen_lamp.jpg
[CLIPFilter] mask=170, best_label='green_tomato', score=0.2524, time=0.01s
[CLIPFilter debug] => wrote debug patch: 2022-07-22-16-25-44-48_patch170_a_tomato_with_dark_green_skin.jpg
[CLIPFilter] mask=171, best_label='half_ripe_tomato', score=0.2723, time=0.01s
[CLIPFilter debug] => wrote debug patch: 2022-07-22-16-25-44-48_patch171_a_tomato_at_mid_ripeness.jpg
[CLIPFilter] mask=172, best_label='negative', score=0.2705, time=0.01s
[CLIPFilter debug] => wrote debug patch: 2022-07-22-16-25-44-48_patch172_a_fluorescent_lamp.jpg
[CLIPFilter] mask=173, best_label='negative', score=0.2579, time=0.01s
[CLIPFilter debug] => wrote debug patch: 2022-07-22-16-25-44-48_patch173_a_fluorescent_lamp.jpg
[CLIPFilter] mask=174, best_label='negative', score=0.2627, time=0.01s
[CLIPFilter debug] => wrote debug patch: 2022-07-22-16-25-44-48_patch174_a_tomato_main_shoot_axis.jpg
[CLIPFilter] mask=175, best_label='negative', score=0.2539, time=0.01s
[CLIPFilter debug] => wrote debug patch: 2022-07-22-16-25-44-48_patch175_a_fluorescent_lamp.jpg
[CLIPFilter] mask=176, best_label='negative', score=0.2629, time=0.01s
[CLIPFilter debug] => wrote debug patch: 2022-07-22-16-25-44-48_patch176_a_shading_screen.jpg
[CLIPFilter] mask=177, best_label='negative', score=0.2682, time=0.01s
[CLIPFilter debug] => wrote debug patch: 2022-07-22-16-25-44-48_patch177_a_fan_blade.jpg
[CLIPFilter] mask=178, best_label='negative', score=0.2670, time=0.01s
[CLIPFilter debug] => wrote debug patch: 2022-07-22-16-25-44-48_patch178_a_fluorescent_lamp.jpg
[CLIPFilter] mask=179, best_label='negative', score=0.2490, time=0.01s
[CLIPFilter debug] => wrote debug patch: 2022-07-22-16-25-44-48_patch179_a_thermal_screen.jpg
[CLIPFilter] mask=180, best_label='negative', score=0.2643, time=0.01s
[CLIPFilter debug] => wrote debug patch: 2022-07-22-16-25-44-48_patch180_a_thermal_screen.jpg
[CLIPFilter] mask=181, best_label='negative', score=0.2809, time=0.01s
[CLIPFilter debug] => wrote debug patch: 2022-07-22-16-25-44-48_patch181_a_fluorescent_lamp.jpg
[CLIPFilter] mask=182, best_label='negative', score=0.2770, time=0.01s
[CLIPFilter debug] => wrote debug patch: 2022-07-22-16-25-44-48_patch182_a_fluorescent_lamp.jpg
[CLIPFilter] mask=183, best_label='negative', score=0.2645, time=0.01s
[CLIPFilter debug] => wrote debug patch: 2022-07-22-16-25-44-48_patch183_a_fluorescent_lamp.jpg
[CLIPFilter] mask=184, best_label='negative', score=0.2497, time=0.01s
[CLIPFilter debug] => wrote debug patch: 2022-07-22-16-25-44-48_patch184_a_support_post.jpg
[CLIPFilter] mask=185, best_label='half_ripe_tomato', score=0.2998, time=0.01s
[CLIPFilter debug] => wrote debug patch: 2022-07-22-16-25-44-48_patch185_a_tomato_at_mid_ripeness.jpg
[CLIPFilter] mask=186, best_label='half_ripe_tomato', score=0.2957, time=0.01s
[CLIPFilter debug] => wrote debug patch: 2022-07-22-16-25-44-48_patch186_a_tomato_at_mid_ripeness.jpg
[CLIPFilter] mask=187, best_label='negative', score=0.2623, time=0.01s
[CLIPFilter debug] => wrote debug patch: 2022-07-22-16-25-44-48_patch187_a_plastic_gutter.jpg
[CLIPFilter] mask=188, best_label='green_tomato', score=0.2848, time=0.01s
[CLIPFilter debug] => wrote debug patch: 2022-07-22-16-25-44-48_patch188_a_tomato_with_dark_green_skin.jpg
[CLIPFilter] mask=189, best_label='negative', score=0.2478, time=0.01s
[CLIPFilter debug] => wrote debug patch: 2022-07-22-16-25-44-48_patch189_a_thermal_screen.jpg
[CLIPFilter] mask=190, best_label='negative', score=0.2563, time=0.01s
[CLIPFilter debug] => wrote debug patch: 2022-07-22-16-25-44-48_patch190_a_fan_blade.jpg
[CLIPFilter] mask=191, best_label='negative', score=0.2754, time=0.01s
[CLIPFilter debug] => wrote debug patch: 2022-07-22-16-25-44-48_patch191_a_fan_blade.jpg
[CLIPFilter] mask=192, best_label='negative', score=0.2725, time=0.01s
[CLIPFilter debug] => wrote debug patch: 2022-07-22-16-25-44-48_patch192_a_floor_tile.jpg
[CLIPFilter] mask=193, best_label='negative', score=0.2607, time=0.01s
[CLIPFilter debug] => wrote debug patch: 2022-07-22-16-25-44-48_patch193_a_fluorescent_lamp.jpg
[CLIPFilter] mask=194, best_label='negative', score=0.2821, time=0.01s
[CLIPFilter debug] => wrote debug patch: 2022-07-22-16-25-44-48_patch194_a_drip_emitter.jpg
[CLIPFilter] mask=195, best_label='ripe_tomato', score=0.2993, time=0.01s
[CLIPFilter debug] => wrote debug patch: 2022-07-22-16-25-44-48_patch195_a_red_tomato_on_a_green_vine.jpg
[CLIPFilter] mask=196, best_label='negative', score=0.2812, time=0.01s
[CLIPFilter debug] => wrote debug patch: 2022-07-22-16-25-44-48_patch196_a_greenhouse_crossbeam.jpg
[CLIPFilter] mask=197, best_label='green_tomato', score=0.2639, time=0.01s
[CLIPFilter debug] => wrote debug patch: 2022-07-22-16-25-44-48_patch197_a_green_tomato_in_greenhouse_row.jpg
[CLIPFilter] mask=198, best_label='negative', score=0.2740, time=0.01s
[CLIPFilter debug] => wrote debug patch: 2022-07-22-16-25-44-48_patch198_a_greenhouse_crossbeam.jpg
[CLIPFilter] mask=199, best_label='negative', score=0.2882, time=0.01s
[CLIPFilter debug] => wrote debug patch: 2022-07-22-16-25-44-48_patch199_a_greenhouse_crossbeam.jpg
[CLIPFilter] mask=200, best_label='ripe_tomato', score=0.2966, time=0.01s
[CLIPFilter debug] => wrote debug patch: 2022-07-22-16-25-44-48_patch200_a_bright_scarlet_tomato.jpg
[CLIPFilter] mask=201, best_label='negative', score=0.2585, time=0.01s
[CLIPFilter debug] => wrote debug patch: 2022-07-22-16-25-44-48_patch201_a_greenhouse_crossbeam.jpg
[CLIPFilter] mask=202, best_label='negative', score=0.2618, time=0.01s
[CLIPFilter debug] => wrote debug patch: 2022-07-22-16-25-44-48_patch202_a_tomato_root_collar_region.jpg
[CLIPFilter] mask=203, best_label='negative', score=0.2938, time=0.01s
[CLIPFilter debug] => wrote debug patch: 2022-07-22-16-25-44-48_patch203_a_lateral_tomato_branch.jpg
[CLIPFilter] mask=204, best_label='negative', score=0.2749, time=0.01s
[CLIPFilter debug] => wrote debug patch: 2022-07-22-16-25-44-48_patch204_a_fan_blade.jpg
[CLIPFilter] mask=205, best_label='negative', score=0.2515, time=0.01s
[CLIPFilter debug] => wrote debug patch: 2022-07-22-16-25-44-48_patch205_a_PVC_pipe.jpg
[CLIPFilter] mask=206, best_label='ripe_tomato', score=0.3510, time=0.01s
[CLIPFilter debug] => wrote debug patch: 2022-07-22-16-25-44-48_patch206_a_glossy_red_tomato.jpg
[CLIPFilter] mask=207, best_label='tomato_leaf', score=0.3162, time=0.01s
[CLIPFilter debug] => wrote debug patch: 2022-07-22-16-25-44-48_patch207_a_glandular_tomato_leaf_surface.jpg
[CLIPFilter] mask=208, best_label='green_tomato', score=0.2830, time=0.01s
[CLIPFilter debug] => wrote debug patch: 2022-07-22-16-25-44-48_patch208_an_immature_green_tomato.jpg
[CLIPFilter] mask=209, best_label='negative', score=0.2809, time=0.01s
[CLIPFilter debug] => wrote debug patch: 2022-07-22-16-25-44-48_patch209_a_plastic_tubing.jpg
[CLIPFilter] mask=210, best_label='negative', score=0.2763, time=0.01s
[CLIPFilter debug] => wrote debug patch: 2022-07-22-16-25-44-48_patch210_a_plastic_tubing.jpg
[CLIPFilter] mask=211, best_label='tomato_leaf', score=0.2926, time=0.01s
[CLIPFilter debug] => wrote debug patch: 2022-07-22-16-25-44-48_patch211_a_tomato_leaf_on_greenhouse_vine.jpg
[CLIPFilter] mask=212, best_label='negative', score=0.2706, time=0.01s
[CLIPFilter debug] => wrote debug patch: 2022-07-22-16-25-44-48_patch212_a_white_board.jpg
[CLIPFilter] mask=213, best_label='negative', score=0.2888, time=0.01s
[CLIPFilter debug] => wrote debug patch: 2022-07-22-16-25-44-48_patch213_a_metal_stake.jpg
[CLIPFilter] mask=214, best_label='negative', score=0.3080, time=0.01s
[CLIPFilter debug] => wrote debug patch: 2022-07-22-16-25-44-48_patch214_a_support_wire.jpg
[CLIPFilter] mask=215, best_label='negative', score=0.3007, time=0.01s
[CLIPFilter debug] => wrote debug patch: 2022-07-22-16-25-44-48_patch215_a_tomato_cane.jpg
[CLIPFilter] mask=216, best_label='negative', score=0.2801, time=0.01s
[CLIPFilter debug] => wrote debug patch: 2022-07-22-16-25-44-48_patch216_an_electrical_cable.jpg
[CLIPFilter] mask=217, best_label='negative', score=0.2717, time=0.01s
[CLIPFilter debug] => wrote debug patch: 2022-07-22-16-25-44-48_patch217_a_zip_tie.jpg
[CLIPFilter] mask=218, best_label='negative', score=0.2645, time=0.01s
[CLIPFilter debug] => wrote debug patch: 2022-07-22-16-25-44-48_patch218_a_PVC_pipe.jpg
[CLIPFilter] mask=219, best_label='negative', score=0.2867, time=0.01s
[CLIPFilter debug] => wrote debug patch: 2022-07-22-16-25-44-48_patch219_a_metal_stake.jpg
[CLIPFilter] mask=220, best_label='tomato_leaf', score=0.3175, time=0.01s
[CLIPFilter debug] => wrote debug patch: 2022-07-22-16-25-44-48_patch220_a_tomato_leaf_in_greenhouse_lighting.jpg
[CLIPFilter] mask=221, best_label='ripe_tomato', score=0.3379, time=0.01s
[CLIPFilter debug] => wrote debug patch: 2022-07-22-16-25-44-48_patch221_a_well-ripened_tomato_on_branch.jpg
[CLIPFilter] mask=222, best_label='green_tomato', score=0.2539, time=0.01s
[CLIPFilter debug] => wrote debug patch: 2022-07-22-16-25-44-48_patch222_a_cluster_of_firm_green_spheres.jpg
[CLIPFilter] mask=223, best_label='negative', score=0.2390, time=0.01s
[CLIPFilter debug] => wrote debug patch: 2022-07-22-16-25-44-48_patch223_a_reflection_on_the_floor.jpg
[CLIPFilter] mask=224, best_label='negative', score=0.2558, time=0.01s
[CLIPFilter debug] => wrote debug patch: 2022-07-22-16-25-44-48_patch224_an_S-hook.jpg
[CLIPFilter] mask=225, best_label='green_tomato', score=0.2777, time=0.01s
[CLIPFilter debug] => wrote debug patch: 2022-07-22-16-25-44-48_patch225_a_green_tomato_under_overhead_netting.jpg
[CLIPFilter] mask=226, best_label='half_ripe_tomato', score=0.2793, time=0.01s
[CLIPFilter debug] => wrote debug patch: 2022-07-22-16-25-44-48_patch226_a_tomato_with_red_shoulders.jpg
[CLIPFilter] mask=227, best_label='negative', score=0.2832, time=0.01s
[CLIPFilter debug] => wrote debug patch: 2022-07-22-16-25-44-48_patch227_a_greenhouse_crossbeam.jpg
[CLIPFilter] mask=228, best_label='negative', score=0.2529, time=0.01s
[CLIPFilter debug] => wrote debug patch: 2022-07-22-16-25-44-48_patch228_a_PVC_pipe.jpg
[CLIPFilter] mask=229, best_label='negative', score=0.2843, time=0.01s
[CLIPFilter debug] => wrote debug patch: 2022-07-22-16-25-44-48_patch229_a_threaded_rod.jpg
[CLIPFilter] mask=230, best_label='negative', score=0.2989, time=0.01s
[CLIPFilter debug] => wrote debug patch: 2022-07-22-16-25-44-48_patch230_a_support_wire.jpg
[CLIPFilter] mask=231, best_label='ripe_tomato', score=0.3492, time=0.01s
[CLIPFilter debug] => wrote debug patch: 2022-07-22-16-25-44-48_patch231_a_red_tomato_cluster_in_greenhouse.jpg
[CLIPFilter] mask=232, best_label='green_tomato', score=0.2751, time=0.01s
[CLIPFilter debug] => wrote debug patch: 2022-07-22-16-25-44-48_patch232_a_green_tomato_hide_under_leaves.jpg
[CLIPFilter] mask=233, best_label='negative', score=0.2576, time=0.01s
[CLIPFilter debug] => wrote debug patch: 2022-07-22-16-25-44-48_patch233_a_greenhouse_crossbeam.jpg
[CLIPFilter] mask=234, best_label='green_tomato', score=0.2843, time=0.01s
[CLIPFilter debug] => wrote debug patch: 2022-07-22-16-25-44-48_patch234_a_green_tomato_under_foliage.jpg
[CLIPFilter] mask=235, best_label='green_tomato', score=0.2752, time=0.01s
[CLIPFilter debug] => wrote debug patch: 2022-07-22-16-25-44-48_patch235_a_green_tomato_in_lower_truss.jpg
[CLIPFilter] mask=236, best_label='negative', score=0.2716, time=0.01s
[CLIPFilter debug] => wrote debug patch: 2022-07-22-16-25-44-48_patch236_a_plastic_tubing.jpg
[CLIPFilter] mask=237, best_label='negative', score=0.2457, time=0.01s
[CLIPFilter debug] => wrote debug patch: 2022-07-22-16-25-44-48_patch237_a_fluorescent_lamp.jpg
[clip_filter] => classification done, now final label filter...
[visualization] => building 'pre' 2x2 composite (sam2) ...
  => [build_composite_for_masks] building annotated overlay...
  => [build_composite_for_masks] building random color + masked array...
  => [build_composite_for_masks] building 2x2 now...
[visualization] => building 'post' 2x2 composite (clip) ...
  => [build_composite_for_masks] building annotated overlay...
  => [build_composite_for_masks] building random color + masked array...
  => [build_composite_for_masks] building 2x2 now...
[visualization] => generating panoptic final image with detectron2 Visualizer...
[visualization] => wrote final single overlay => demos/tomato/output/2022-07-22-16-25-44-48-final.jpg
[process_folder] => wrote JSON => demos/tomato/output/2022-07-22-16-25-44-48.json
[visualization] => wrote summary => demos/tomato/output/2022-07-22-16-25-44-48_summary.jpg
[process_folder] => done with image.

