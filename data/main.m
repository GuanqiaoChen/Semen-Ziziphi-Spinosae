close all; clear; clc;

%% ===================== 读取高光谱数据 =====================
hcube = hypercube('F:\cut_szr\7-2.hdr');
data_cube = hcube.DataCube; % [rows, cols, bands]
wavelengths = hcube.Wavelength(:);
[rows, cols, bands] = size(data_cube);
fprintf('数据立方体尺寸: 行=%d, 列=%d, 波段=%d\n', rows, cols, bands);

%% ===================== 伪彩色用于手动选区 =====================
middle_band = round(bands/2);
band1_idx = max(1, round(bands * 0.2));
band2_idx = middle_band;
band3_idx = min(bands, round(bands * 0.8));

rgb_image = zeros(rows, cols, 3);
rgb_image(:, :, 1) = squeeze(data_cube(:, :, band1_idx));
rgb_image(:, :, 2) = squeeze(data_cube(:, :, band2_idx));
rgb_image(:, :, 3) = squeeze(data_cube(:, :, band3_idx));

for i = 1:3
    min_val = min(rgb_image(:, :, i), [], 'all');
    max_val = max(rgb_image(:, :, i), [], 'all');
    if max_val > min_val
        rgb_image(:, :, i) = (rgb_image(:, :, i) - min_val) / (max_val - min_val);
    end
end

figure; imshow(rgb_image);
title('请在图像上拖动鼠标选择一个矩形区域');
h = imrect; wait(h);
position = getPosition(h);

min_col = max(1, round(position(1)));
min_row = max(1, round(position(2)));
max_col = min(cols, round(position(1) + position(3) - 1));
max_row = min(rows, round(position(2) + position(4) - 1));

bbox_rows = max_row - min_row + 1;
bbox_cols = max_col - min_col + 1;
fprintf('选择的处理区域: 行=%d-%d, 列=%d-%d (尺寸 %d x %d)\n', min_row, max_row, min_col, max_col, bbox_rows, bbox_cols);

bbox_data = data_cube(min_row:max_row, min_col:max_col, :);

%% ===================== ROI内伪彩色与分割 =====================
rgb_bbox = zeros(bbox_rows, bbox_cols, 3);
rgb_bbox(:, :, 1) = squeeze(bbox_data(:, :, band1_idx));
rgb_bbox(:, :, 2) = squeeze(bbox_data(:, :, band2_idx));
rgb_bbox(:, :, 3) = squeeze(bbox_data(:, :, band3_idx));
for i = 1:3
    ch = rgb_bbox(:,:,i);
    mn = min(ch(:)); mx = max(ch(:));
    if mx > mn
        rgb_bbox(:,:,i) = (ch - mn)/(mx - mn);
    end
end
figure; imshow(rgb_bbox); title('ROI 伪彩色图像');

gray_image = rgb2gray(rgb_bbox);
threshold = graythresh(gray_image);
binary_image = imbinarize(gray_image, threshold);
se = strel('square', 2);
binary_image = imclose(binary_image, se);
binary_image = imopen(binary_image, se);

cc = bwconncomp(binary_image);
num_objects = cc.NumObjects;
if num_objects == 0
    warning('未检测到物体，降低阈值重试');
    binary_image = imbinarize(gray_image, threshold*0.5);
    binary_image = imclose(binary_image, se);
    binary_image = imopen(binary_image, se);
    cc = bwconncomp(binary_image);
    num_objects = cc.NumObjects;
end
if num_objects == 0
    error('无法检测到任何物体');
end

labeled_image = labelmatrix(cc);
figure; imshow(label2rgb(labeled_image, 'hsv', 'k', 'shuffle'));
title(sprintf('分割结果：%d 个物体', num_objects));

stats = regionprops(cc, 'Centroid', 'BoundingBox', 'Area');
hold on;
for i = 1:num_objects
    cen = stats(i).Centroid;
    text(cen(1), cen(2), num2str(i), 'Color','w','FontSize',12,'FontWeight','bold',...
        'HorizontalAlignment','center');
end
hold off;

%% ===================== 参数与保存路径 =====================
patchSize = 32;       % <<< 固定正方形 patch 尺寸
target_folder = 'F:\szr_c_cube\7-2';
if ~exist(target_folder, 'dir')
    mkdir(target_folder);
end

mean_spectra = zeros(num_objects, bands);

%% ===================== 主循环：固定尺寸 patch + 平均光谱 =====================
half = floor(patchSize / 2);  % 用于质心居中

for i = 1:num_objects
    obj_mask = (labeled_image == i);                    % ROI 坐标系下的二值掩膜
    cen = stats(i).Centroid;                            % 质心 (col, row)
    cen_row = round(cen(2));
    cen_col = round(cen(1));
    
    % 以质心为中心，定义 patchSize x patchSize 的窗口（可能越界）
    r1 = cen_row - half;
    r2 = cen_row + half - mod(patchSize+1,2);  % 保证正好 patchSize 行
    c1 = cen_col - half;
    c2 = cen_col + half - mod(patchSize+1,2);
    
    % 处理越界（用 padarray 补 0）
    padTop    = max(0, 1 - r1);
    padBottom = max(0, r2 - bbox_rows);
    padLeft   = max(0, 1 - c1);
    padRight  = max(0, c2 - bbox_cols);
    
    r1 = max(r1, 1);  r2 = min(r2, bbox_rows);
    c1 = max(c1, 1);  c2 = min(c2, bbox_cols);
    
    % 裁剪
    crop_cube = bbox_data(r1:r2, c1:c2, :);
    crop_mask = obj_mask(r1:r2, c1:c2);
    
    % 补齐到 patchSize
    if any([padTop padBottom padLeft padRight])
        crop_cube = padarray(crop_cube, [padTop padLeft 0], 0, 'pre');
        crop_cube = padarray(crop_cube, [padBottom padRight 0], 'post');
        crop_mask = padarray(crop_mask, [padTop padLeft], false, 'pre');
        crop_mask = padarray(crop_mask, [padBottom padRight], false, 'post');
    end
    
    % 确保尺寸正好是 patchSize（理论上已经是）
    crop_cube = crop_cube(1:patchSize, 1:patchSize, :);
    crop_mask = crop_mask(1:patchSize, 1:patchSize);
    
    % 背景清零（mask 外为 0）
    mask3d = repmat(crop_mask, [1 1 bands]);
    crop_cube(~mask3d) = 0;
    
    % ===== 基于此固定 patch 计算平均光谱（只平均前景像素）=====
    foreground_pixels = sum(crop_mask(:));
    if foreground_pixels == 0
        warning('物体 %d 无前景像素', i);
        mean_spectra(i,:) = 0;
    else
        for b = 1:bands
            ch = crop_cube(:,:,b);
            mean_spectra(i,b) = sum(ch(crop_mask)) / foreground_pixels;
        end
    end
    fprintf('物体 %d: 前景像素 %d，平均光谱已计算\n', i, foreground_pixels);
    
    % 转为 PyTorch 常用格式 [C, H, W]
    patch_chw = permute(crop_cube, [3 1 2]);  % [bands, patchSize, patchSize]
    patch_hwc = crop_cube;                   % [patchSize, patchSize, bands]
    
    % 元信息
    meta.object_id = i;
    meta.wavelengths = wavelengths;
    meta.patch_size = patchSize;
    meta.centroid_in_bbox = [cen_row, cen_col];
    meta.centroid_in_full = [min_row + cen_row - 1, min_col + cen_col - 1];
    meta.note = 'fixed-size centroid patch + background zeroed';
    
    % 保存 .mat
    mat_filename = fullfile(target_folder, sprintf('%d.mat', i));
    save(mat_filename, 'patch_chw', 'patch_hwc', 'crop_mask', 'meta', '-v7.3');
    fprintf('物体 %d 已保存固定 patch %s (%dx%dx%d)\n', i, mat_filename, patchSize, patchSize, bands);
end

%% ===================== 绘制与保存平均光谱 =====================
figure; hold on; grid on;
for i = 1:num_objects
    plot(wavelengths, mean_spectra(i,:), 'LineWidth', 1.5);
end
xlabel('波长 (nm)'); ylabel('反射率/信号值');
title('各物体的平均光谱（基于固定 patch）');
legend(arrayfun(@(x) sprintf('物体 %d', x), 1:num_objects, 'UniformOutput', false));
hold off;

for i = 1:num_objects
    csv_filename = fullfile(target_folder, sprintf('%d.csv', i));
    writematrix([wavelengths, mean_spectra(i,:)'], csv_filename);
end
fprintf('平均光谱 CSV 已全部保存。\n');
fprintf('全部完成：固定 %d×%d patch + 对应平均光谱。\n', patchSize, patchSize);