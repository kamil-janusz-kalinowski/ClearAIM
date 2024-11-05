clc; clear; close all;

addpath(genpath("./"));

paths = findImageMaskPairs("./Materials", "./Results");
paths = sort_image_mask_struct(paths);


figure;
for ind = 1 : length(paths)
    % Wczytaj obraz i maskę
    path_img = paths(ind).image_path;
    path_mask = paths(ind).mask_path;
    
    img = imread(path_img);
    mask = imread(path_mask);
    
    binary_mask = processMask(mask, size(img, [1, 2]));
    
    % Konwersja obrazu do skali szarości (jeśli jest kolorowy)
    if size(img, 3) == 3
        img_gray = rgb2gray(img);
    else
        img_gray = img;
    end
    
    % Obliczenie histogramu obrazu
    threshold = graythresh(img_gray(binary_mask)); % Wyznaczenie progu (lub własny próg)
    threshold_value = threshold * 255;
    
    % Wyświetlanie histogramu
    subplot(1, 2, 1);
    imhist(img_gray(binary_mask));
    hold on;
    line([threshold_value threshold_value], ylim, 'Color', 'r', 'LineWidth', 2);
    hold off;
    title('Histogram z progiem');
    
    % Maski dla pikseli powyżej i poniżej progu
    above_threshold = img_gray > threshold_value & binary_mask;
    below_threshold = img_gray <= threshold_value & binary_mask;
    
    % Tworzenie kolorowego obrazu dla wizualizacji
    img_rgb = repmat(img_gray, [1, 1, 3]); % Tworzenie obrazu RGB z obrazu w skali szarości
    
    % Nakładanie półprzezroczystych kolorów
    alpha = 0.5; % Poziom przezroczystości
    img_rgb(:,:,1) = img_rgb(:,:,1) + uint8(255 * alpha * above_threshold); % Czerwony dla powyżej progu
    img_rgb(:,:,3) = img_rgb(:,:,3) + uint8(255 * alpha * below_threshold); % Niebieski dla poniżej progu
    
    % Wyświetlanie obrazu z podświetlonymi pikselami
    subplot(1, 2, 2);
    imshow(img_rgb);
    title('Obraz z podświetleniem progów');
    
    % Wyświetlanie postępu
    progress = (ind / length(paths)) * 100;
    title(['Processing... ', num2str(progress, '%.1f'), '%']);
    
    pause(0.01); % Krótka pauza dla wizualizacji
end



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
