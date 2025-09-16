# Category Label Mapping Function Description

## Overview

The `translate_user_classes` function now supports generic label mapping functionality, allowing users to use more natural language to describe object categories to be detected.

## Features

### 1. Generic Label Mapping
When users input queries containing generic labels, the system automatically expands them to all related specific categories:

- `"find fruit"` → detects `["apple", "orange", "banana"]`
- `"look for vehicles"` → detects `["car", "truck", "bus", "motorcycle", "bicycle", "airplane", "train", "boat"]`
- `"detect animals"` → detects `["person", "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe"]`

### 2. Multi-Label Combination
Supports using multiple generic labels simultaneously:

- `"find fruit and vegetables"` → detects `["apple", "orange", "banana", "broccoli", "carrot"]`
- `"detect pets and electronics"` → detects `["cat", "dog", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "refrigerator"]`

### 3. Backward Compatibility
The original direct category name matching and AI intelligent translation functions remain unchanged.

## Supported Generic Labels

### Food Related
- `fruit` → `["apple", "orange", "banana"]`
- `vegetable` → `["broccoli", "carrot"]`
- `food` → `["banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake"]`
- `drink` → `["bottle", "wine glass", "cup"]`

### Transportation
- `vehicle` → `["car", "truck", "bus", "motorcycle", "bicycle", "airplane", "train", "boat"]`
- `transportation` → includes all transportation and sports equipment
- `road_vehicle` → `["car", "truck", "bus", "motorcycle", "bicycle"]`
- `air_vehicle` → `["airplane"]`
- `water_vehicle` → `["boat"]`
- `rail_vehicle` → `["train"]`

### Animals
- `animal` → `["person", "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe"]`
- `pet` → `["cat", "dog"]`
- `wildlife` → `["bird", "elephant", "bear", "zebra", "giraffe"]`
- `farm_animal` → `["horse", "sheep", "cow"]`

### Electronic Devices
- `electronic` → `["tv", "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "refrigerator"]`
- `appliance` → `["microwave", "oven", "toaster", "refrigerator", "sink", "hair drier"]`
- `kitchen` → includes kitchen-related equipment and utensils
- `office` → `["laptop", "mouse", "remote", "keyboard", "cell phone"]`

### Other Categories
- `furniture` → `["chair", "couch", "bed", "dining table"]`
- `sport` → includes all sports equipment
- `bag` → `["backpack", "handbag", "suitcase"]`
- `indoor` → indoor items
- `outdoor` → outdoor items

## Usage Examples

### Voice Command Examples
```
"find fruit for me"
"detect vehicles in the scene"
"look for animals"
"find food and drinks"
"search for electronics"
```

### Text Command Examples
```
find fruit
look for vehicles
detect animals
find food
search for electronics
find fruit and vegetables
detect pets
find transportation
```

## Technical Implementation

### Workflow
1. User inputs a query
2. System first checks generic label mappings in `category_labels.json`
3. If matching labels are found, directly returns the corresponding specific category list
4. If not found, continues using the original AI intelligent translation function
5. Finally validates that all categories exist in the YOLO-supported category list

### File Structure
- `category_labels.json` - mapping file from generic labels to specific categories
- `ada.py` - modified `translate_user_classes` function

### Extensibility
New generic label mappings can be added by modifying the `category_labels.json` file without changing the code.

## Notes

1. Generic label matching is case-insensitive
2. System automatically removes duplicate categories
3. Only returns valid categories that exist in the YOLO category list
4. If generic label mapping fails, automatically falls back to the original AI translation function 