from flask import Flask, render_template, request, send_file
from flask_cors import CORS
import cv2
import numpy as np
import io
import os

app = Flask(__name__)
CORS(app)

# Create uploads folder if it doesn't exist
UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
def process_image(image, style):
    """Process the image with the selected style"""
    try:
        # Ensure proper color format
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        elif image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
            
        if style == 'ghibli':
            # 1. Initial size check and resize
            height, width = image.shape[:2]
            max_size = 1024
            if max(height, width) > max_size:
                scale = max_size / max(height, width)
                new_width = int(width * scale)
                new_height = int(height * scale)
                image = cv2.resize(image, (new_width, new_height))
            
            # 2. Initial detail preservation
            # Use bilateral filter to preserve edges while smoothing
            smooth = cv2.bilateralFilter(image, 9, 75, 75)
            
            # 3. Edge detection for strong lines
            edges = cv2.Canny(smooth, 50, 150)
            edges = cv2.dilate(edges, None)
            
            # 4. Convert to float32 for processing
            img_float = smooth.astype(np.float32) / 255.0
            
            # 5. Military style color grading
            # Convert to LAB for precise color control
            lab = cv2.cvtColor(img_float, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            
            # Enhance contrast in luminance
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            l = clahe.apply((l * 255).astype(np.uint8)).astype(np.float32) / 255.0
            
            # Adjust colors for military look
            a = a * 0.9  # Reduce red-green
            b = b * 0.85  # Reduce blue-yellow
            
            # Merge back
            lab = cv2.merge([l, a, b])
            img_float = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
            
            # 6. Military uniform color enhancement
            hsv = cv2.cvtColor(img_float, cv2.COLOR_BGR2HSV)
            h, s, v = cv2.split(hsv)
            
            # Adjust saturation and value
            s = s * 0.9  # Slightly desaturate for uniform look
            v = v * 1.1  # Increase brightness slightly
            
            # Merge and clip
            hsv = cv2.merge([h, s, v])
            hsv = np.clip(hsv, 0, 1)
            img_float = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
            
            # 7. Add subtle light effect
            light = cv2.GaussianBlur(img_float, (0, 0), 10)
            img_float = cv2.addWeighted(img_float, 0.85, light, 0.15, 0)
            
            # 8. Enhance contrast
            # Use modified sigmoid for military portrait look
            img_float = 1 / (1 + np.exp(-10 * (img_float - 0.5)))
            
            # 9. Add strong line art effect
            result = (img_float * 255).astype(np.uint8)
            
            # Apply edge-preserving filter for clean look
            result = cv2.edgePreservingFilter(result, flags=1, sigma_s=60, sigma_r=0.4)
            
            # 10. Enhance edges
            edge_mask = cv2.GaussianBlur(edges.astype(np.float32) / 255, (0, 0), 1)
            result = cv2.addWeighted(result, 1, (edge_mask[:, :, None] * result).astype(np.uint8), 0.1, 0)
            
            # 11. Final color adjustments
            result = cv2.cvtColor(result, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(result)
            
            # Enhance shadows and highlights
            l = clahe.apply(l)
            
            # Military green tint
            a = cv2.convertScaleAbs(a, alpha=0.95, beta=-2)  # Reduce red
            b = cv2.convertScaleAbs(b, alpha=0.95, beta=-1)  # Reduce yellow
            
            # Final merge
            result = cv2.merge([l, a, b])
            result = cv2.cvtColor(result, cv2.COLOR_LAB2BGR)
            
            # 12. Final detail enhancement
            sharp_kernel = np.array([[0,-1,0], [-1,5,-1], [0,-1,0]])
            result = cv2.filter2D(result, -1, sharp_kernel)
            
            # 13. Resize back if needed
            if max(height, width) > max_size:
                result = cv2.resize(result, (width, height))
        else:
            # Apply anime-like effect
            # Reduce noise and preserve edges
            img_color = cv2.bilateralFilter(image, 5, 50, 50)
            
            # Convert to grayscale for edge detection
            img_gray = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)
            
            # Edge detection
            edges = cv2.adaptiveThreshold(img_gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 9, 2)
            
            # Combine color and edges
            result = cv2.bitwise_and(img_color, img_color, mask=edges)
        
        return result
    except Exception as e:
        print(f"Error processing image: {str(e)}")
        return image  # Return original image if processing fails



def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in {'png', 'jpg', 'jpeg', 'gif'}

def apply_anime_style(image, settings=None):
    if settings is None:
        settings = {
            'edge_strength': 1.5,
            'color_strength': 1.4,
            'smoothing': 0.8,
            'contrast': 1.4
        }

    try:
        # Convert to RGB if needed
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)

        # Edge detection and enhancement
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        edge = cv2.Canny(gray, 100, 200)
        edge = cv2.dilate(edge, None)
        edge = cv2.GaussianBlur(edge, (3, 3), 0)
        edge = edge.astype(float) / 255.0 * settings['edge_strength']

        # Color quantization
        img_small = cv2.resize(image, None, fx=0.5, fy=0.5)
        Z = img_small.reshape((-1, 3))
        Z = np.float32(Z)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        K = 8
        _, labels, centers = cv2.kmeans(Z, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
        centers = np.uint8(centers)
        res = centers[labels.flatten()]
        quantized = res.reshape((img_small.shape))
        quantized = cv2.resize(quantized, (image.shape[1], image.shape[0]))

        # Color enhancement
        hsv = cv2.cvtColor(quantized, cv2.COLOR_RGB2HSV).astype(np.float32)
        hsv[:, :, 1] = hsv[:, :, 1] * settings['color_strength']
        hsv = np.clip(hsv, 0, 255).astype(np.uint8)
        color_enhanced = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

        # Apply edge overlay
        edge = cv2.cvtColor(edge.astype(np.float32), cv2.COLOR_GRAY2RGB)
        result = cv2.addWeighted(color_enhanced.astype(float), 1.0, (1 - edge) * 255, 1.0, 0)
        result = np.clip(result, 0, 255).astype(np.uint8)

        # Final smoothing
        if settings['smoothing'] > 0:
            result = cv2.GaussianBlur(result, (3, 3), settings['smoothing'])

        return result

    except Exception as e:
        print(f"Error in anime processing: {str(e)}")
        return image

def apply_ghibli_style(image, settings=None):
    if settings is None:
        settings = {
            'edge_strength': 1.2,
            'color_strength': 1.3,
            'smoothing': 1.0,
            'contrast': 1.2
        }

    try:
        # Convert to RGB if needed
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)

        # Step 1: Edge preservation and enhancement
        edge_kernel = np.array([[-1,-1,-1],
                               [-1, 9,-1],
                               [-1,-1,-1]]) * settings['edge_strength']
        edge_enhanced = cv2.filter2D(image, -1, edge_kernel)

        # Step 2: Bilateral filtering for smooth areas while preserving edges
        smooth = cv2.bilateralFilter(edge_enhanced, 9, 75, 75)

        # Step 3: Color enhancement
        # Convert to LAB color space
        lab = cv2.cvtColor(smooth, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)

        # Enhance contrast
        clahe = cv2.createCLAHE(clipLimit=settings['contrast'], tileGridSize=(8,8))
        l = clahe.apply(l)

        # Merge channels
        lab = cv2.merge([l, a, b])
        color_enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)

        # Step 4: Ghibli-style color grading
        hsv = cv2.cvtColor(color_enhanced, cv2.COLOR_RGB2HSV).astype(np.float32)
        
        # Enhance saturation
        hsv[:, :, 1] = hsv[:, :, 1] * settings['color_strength']
        
        # Adjust brightness
        hsv[:, :, 2] = hsv[:, :, 2] * 1.1
        
        # Clip values
        hsv = np.clip(hsv, 0, 255).astype(np.uint8)
        
        # Convert back to RGB
        result = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

        # Step 5: Final smoothing
        if settings['smoothing'] > 0:
            result = cv2.GaussianBlur(result, (3, 3), settings['smoothing'])

        return result

    except Exception as e:
        print(f"Error in image processing: {str(e)}")
        return image

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/settings', methods=['POST', 'OPTIONS'])
def update_settings():
    if request.method == 'OPTIONS':
        return '', 204
    try:
        settings = request.json
        if not settings:
            return jsonify({'error': 'No settings provided'}), 400
        return jsonify({'status': 'success', 'settings': settings})
    except Exception as e:
        return jsonify({'error': f'Error updating settings: {str(e)}'}), 400

@app.route('/convert', methods=['POST'])
def convert_image():
    try:
        # Check if image was uploaded
        if 'image' not in request.files:
            return 'No image uploaded', 400
            
        file = request.files['image']
        if not file.filename:
            return 'No image selected', 400
            
        # Read image
        img_data = file.read()
        img_array = np.frombuffer(img_data, np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        
        if img is None:
            return 'Invalid image file', 400
            
        # Get style choice
        style = request.form.get('style', 'ghibli')
        
        # Process image
        result = process_image(img, style)
        
        # Convert to bytes
        _, buffer = cv2.imencode('.jpg', result)
        output = io.BytesIO(buffer)
        
        return send_file(
            output,
            mimetype='image/jpeg',
            as_attachment=True,
            download_name='converted.jpg'
        )
        
    except Exception as e:
        return str(e), 500

@app.errorhandler(413)
def request_entity_too_large(error):
    return jsonify({'error': 'File too large. Maximum size is 16MB'}), 413

@app.errorhandler(500)
def internal_error(error):
    return jsonify({'error': 'Internal server error occurred'}), 500

@app.errorhandler(404)
def not_found_error(error):
    return jsonify({'error': 'Resource not found'}), 404

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5000)
