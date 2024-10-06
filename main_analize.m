clc; clear; close all;

addpath(genpath("./"));

paths = findImageMaskPairs("./Materials", "./Results");
paths = sort_image_mask_struct(paths);

%% Show mask on image
for ind = 1 : length(paths)
    path_img = paths(ind).image_path;
    path_mask = paths(ind).mask_path;
    displayImageWithMaskContour(path_img, path_mask)

    progress = (ind / length(paths)) * 100;
    title(['Processing... ', num2str(progress, '%.1f'), '%']);

    pause(0.01)
end

%% Calculate and Display Sample Measurements

RADIUS_DILAT = 20;

web_contrast = zeros(length(paths), 1);
area = zeros(length(paths), 1);
for ind = 1 : length(paths)
    img_origin = imread(paths(ind).image_path);
    img_mask = imread(paths(ind).mask_path);
    img_mask = processMask(img_mask, size(img_origin));

    mean_value_object = mean(img_origin(img_mask(:)));
    mask_background = bwmorph(img_mask, 'dilate', RADIUS_DILAT) & ~img_mask;
    mean_value_background = mean(img_origin(mask_background(:)));
    
    web_contrast(ind) = -weberContrast(mean_value_object, mean_value_background);
    area(ind) = sum(img_mask(:));

    disp("Progress: " + num2str(ind/length(paths) * 100) + "%")
end

figure
plot(area(2:end), 'LineWidth', 2)
title('Area plot')

figure
plot(web_contrast(2:end), 'LineWidth', 2)
title('Weber contrast plot')

%% Making animation

% Assuming the functions findImageMaskPairs, displayImageWithMaskContour, 
% processMask, and weberContrast are already defined.

% Find and sort the image and mask paths
paths = findImageMaskPairs("./Materials", "./Results");
paths = sort_image_mask_struct(paths);

% Preallocate arrays for measurements
web_contrast = zeros(length(paths), 1);
area = zeros(length(paths), 1);

% Create video writer
videoWriter = VideoWriter('animation.avi'); % Name of the output video file
open(videoWriter);

% Create figure for animation
figure;

% Show mask on image and calculate measurements
for ind = 1 : length(paths)
    path_img = paths(ind).image_path;
    path_mask = paths(ind).mask_path;

    % Display the image with mask contours
    subplot(1, 3, 1);
    displayImageWithMaskContour(path_img, path_mask);
    title('Image with Mask Contours');
    
    % Update progress title
    progress = (ind / length(paths)) * 100;
    title(['Processing... ', num2str(progress, '%.1f'), '%']);

    % Calculate Weber contrast and area
    img_origin = imread(path_img);
    img_mask = imread(path_mask);
    img_mask = processMask(img_mask, size(img_origin));

    mean_value_object = mean(img_origin(img_mask(:)));
    mask_background = bwmorph(img_mask, 'dilate', RADIUS_DILAT) & ~img_mask;
    mean_value_background = mean(img_origin(mask_background(:)));

    web_contrast(ind) = -weberContrast(mean_value_object, mean_value_background);
    area(ind) = sum(img_mask(:));

    % Plot area and contrast
    subplot(1, 3, 2);
    plot(1:ind, area(1:ind), 'LineWidth', 2);
    title('Area Plot');
    xlabel('Frame');
    ylabel('Area Value');
    xlim([1 length(paths)]);
    ylim([0 max(area) * 1.1]); % Adjust y-axis for better visibility

    subplot(1, 3, 3);
    plot(1:ind, web_contrast(1:ind), 'LineWidth', 2);
    title('Weber Contrast Plot');
    xlabel('Frame');
    ylabel('Weber Contrast Value');
    xlim([1 length(paths)]);
    ylim([0 max(web_contrast) * 1.1]); % Adjust y-axis for better visibility

    % Capture the frame
    frameData = getframe(gcf);
    writeVideo(videoWriter, frameData);

    % Pause briefly to visualize the process
    pause(0.01);
end

% Close the video writer
close(videoWriter);

disp('Animation completed and saved as animation.avi');


function file_pairs = findImageMaskPairs(folder_images, folder_masks, file_format)
    % Retrieves pairs of image and corresponding mask files from specified folders
    %
    % This function searches for image files in the specified image folder,
    % based on the provided file format, and tries to find corresponding mask
    % files with a "_mask" suffix in the specified mask folder. It returns a
    % structure array with image and mask paths.
    %
    % Parameters:
    %    folder_images (string): Path to the folder containing image files
    %    folder_masks (string): Path to the folder containing mask files
    %    file_format (string): File format for the images (e.g., '*.png')
    %
    % Returns:
    %    file_pairs (struct): Structure containing pairs of image and mask paths
    %
    % Example:
    %    file_pairs = findImageMaskPairs('images_folder', 'masks_folder', '*.jpg');
    
    % Set default file format if not provided
    if nargin < 3
        file_format = '*.png';
    end
    
    % Get a list of files in the image folder with the specified format
    image_files = dir(fullfile(folder_images, file_format));
    
    % Initialize a structure to store results
    file_pairs = struct('image_path', {}, 'mask_path', {});
    
    % Iterate over all image files
    for i = 1:length(image_files)
        % Get the image filename without extension
        [~, filename, ~] = fileparts(image_files(i).name);
        
        % Construct the corresponding mask file path in the second folder
        mask_filename = [filename, '_mask.png'];
        mask_path = fullfile(folder_masks, mask_filename);
        
        % Check if the mask file exists
        if isfile(mask_path)
            % Add the image and mask paths to the structure
            file_pairs(end+1).image_path = fullfile(folder_images, image_files(i).name);
            file_pairs(end).mask_path = mask_path;
        end
    end
end


function binary_mask = processMask(mask, target_size)
    %PROCESSMASK Processes the input mask by converting it to grayscale,
    % resizing it to the target size, and binarizing it.
    %
    % binary_mask = PROCESSMASK(mask, target_size) takes the input mask and
    % processes it by converting it to grayscale (if needed), resizing it to the
    % specified target_size, and binarizing the result.
    %
    % Inputs:
    %   mask - The input mask image.
    %   target_size - The desired size for the output mask [rows, cols].
    %
    % Outputs:
    %   binary_mask - The processed binary mask.

    % If the mask is color, convert it to grayscale
    if size(mask, 3) == 3
        mask = rgb2gray(mask);
    end
    
    % Resize the mask to the target size
    resized_mask = imresize(mask, target_size);
    
    % Binarize the mask
    binary_mask = imbinarize(resized_mask);
end

function displayImageWithMaskContour(image_path, mask_path)
    % displayImageWithMaskContour Displays an image with overlaid mask contours
    %
    % This function loads an original image and a corresponding mask, processes
    % the mask to create a binary version, finds the contours of the mask, and
    % displays the original image with the mask contours overlaid in red.
    %
    % Parameters:
    %    image_path (string): Path to the image file
    %    mask_path (string): Path to the mask file
    %
    % Example:
    %    displayImageWithMaskContour('image.jpg', 'mask.png');
    
    % Load the original image
    original_image = imread(image_path);
    
    % Load the mask
    mask = imread(mask_path);
    
    % Process the mask (using a helper function)
    binary_mask = processMask(mask, [size(original_image, 1), size(original_image, 2)]);
    
    % Find the mask contours
    mask_contours = bwperim(binary_mask);
    
    % Display the original image
    imshow(original_image);
    hold on;
    
    % Overlay the contours on the image (red contour color)
    visboundaries(mask_contours, 'Color', 'r');
    
    hold off;
end



function C = weberContrast(I_sample, I_background)
    %WEBER_CONTRAST Calculates the Weber contrast between a sample and the background
    %
    % C = WEBER_CONTRAST(I_sample, I_background) computes the Weber contrast (C) for a given
    % sample intensity (I_sample) and background intensity (I_background). 
    %
    % The Weber contrast is defined as:
    % C = (I_sample - I_background) / I_background
    %
    % This measure is useful when the background intensity is uniform and the sample's
    % intensity varies slightly from the background.
    %
    % Inputs:
    %   I_sample - Intensity of the sample (scalar or array).
    %   I_background - Intensity of the background (scalar or array of the same size as I_sample).
    %
    % Outputs:
    %   C - The computed Weber contrast (scalar or array).
    %
    % Example:
    %   I_sample = 150;
    %   I_background = 100;
    %   C = weber_contrast(I_sample, I_background);
    %   disp(C);  % Outputs: 0.5
    
    C = (I_sample - I_background) ./ I_background;
end

function sortedStruct = sort_image_mask_struct(inputStruct)
    %SORT_IMAGE_MASK_STRUCT Sorts a structure array based on the numerical values in the image_path field.
    %
    % sortedStruct = SORT_IMAGE_MASK_STRUCT(inputStruct) takes a structure array with fields
    % image_path and mask_path and sorts it according to the numerical values extracted from
    % the image_path field.
    %
    % Inputs:
    %   inputStruct - A structure array with fields 'image_path' and 'mask_path'.
    %
    % Outputs:
    %   sortedStruct - A structure array sorted based on the numerical values in the image_path.

    % Extract image paths
    imagePaths = {inputStruct.image_path};
    
    % Initialize an array to hold numerical values
    nums = zeros(size(imagePaths));

    % Loop through each image path to extract numerical values
    for i = 1:length(imagePaths)
        % Use regexp to find numeric part
        match = regexp(imagePaths{i}, '_(\d+)\.png$', 'tokens'); % Extract numeric part
        if ~isempty(match)
            nums(i) = str2double(match{1}{1}); % Convert to double
        else
            nums(i) = NaN; % Handle cases with no match
        end
    end

    % Sort structure based on the extracted numbers, ignoring NaNs
    [~, sortIdx] = sort(nums); % Sort indices
    sortedStruct = inputStruct(sortIdx);  % Reorder structure based on sorted indices
end
