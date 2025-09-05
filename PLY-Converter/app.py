import os
import uuid
import logging
import time
import sys
from flask import Flask, render_template, request, jsonify, send_file, url_for
from werkzeug.utils import secure_filename
import threading
import traceback

# Configure detailed logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('ply_converter_app.log')
    ]
)

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['OUTPUT_FOLDER'] = 'outputs'
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 500MB max file size

logger = logging.getLogger(__name__)

# Ensure directories exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['OUTPUT_FOLDER'], exist_ok=True)

# Store conversion progress
conversion_progress = {}
conversion_results = {}

ALLOWED_EXTENSIONS = {'ply'}
OUTPUT_FORMATS = ['stl', 'obj', 'glb', '3mf', 'dxf']
SMOOTHING_LEVELS = ['light', 'medium', 'high', 'ultra']

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    logger.info("=== UPLOAD REQUEST RECEIVED ===")
    
    try:
        # Log request details
        logger.info(f"Request method: {request.method}")
        logger.info(f"Request files: {list(request.files.keys())}")
        logger.info(f"Request form: {dict(request.form)}")
        
        if 'file' not in request.files:
            logger.error("No file in request")
            return jsonify({'error': 'No file uploaded'}), 400
        
        file = request.files['file']
        logger.info(f"File received: {file.filename}, size: {file.content_length}")
        
        if file.filename == '':
            logger.error("No file selected")
            return jsonify({'error': 'No file selected'}), 400
        
        if not allowed_file(file.filename):
            logger.error(f"Invalid file type: {file.filename}")
            return jsonify({'error': 'Only PLY files are allowed'}), 400
        
        # Generate unique ID for this conversion
        conversion_id = str(uuid.uuid4())
        logger.info(f"Generated conversion ID: {conversion_id}")
        
        # Save uploaded file
        filename = secure_filename(file.filename)
        input_path = os.path.join(app.config['UPLOAD_FOLDER'], f"{conversion_id}_{filename}")
        
        logger.info(f"Saving file to: {input_path}")
        file.save(input_path)
        
        # Verify file was saved
        if os.path.exists(input_path):
            file_size = os.path.getsize(input_path)
            logger.info(f"✓ File saved successfully: {file_size} bytes")
        else:
            logger.error("✗ File save failed")
            return jsonify({'error': 'File save failed'}), 500
        
        # Get parameters
        output_formats = request.form.getlist('formats')
        if not output_formats:
            output_formats = ['stl']
        
        smoothing_level = request.form.get('smoothing', 'medium')
        if smoothing_level not in SMOOTHING_LEVELS:
            smoothing_level = 'medium'
        
        # Validate output formats
        valid_formats = [fmt for fmt in output_formats if fmt in OUTPUT_FORMATS]
        if not valid_formats:
            logger.error(f"No valid formats: {output_formats}")
            return jsonify({'error': 'Invalid output formats specified'}), 400
        
        logger.info(f"Conversion parameters: formats={valid_formats}, smoothing={smoothing_level}")
        
        # Initialize progress tracking
        conversion_progress[conversion_id] = {
            'status': 'starting',
            'progress': 0,
            'message': 'Initializing smooth surface conversion...',
            'input_file': filename,
            'output_formats': valid_formats,
            'smoothing_level': smoothing_level,
            'created_at': time.time()
        }
        
        logger.info(f"Progress tracking initialized for {conversion_id}")
        
        # Start conversion in background thread
        logger.info("Starting background conversion thread...")
        thread = threading.Thread(
            target=convert_file_async_debug,
            args=(conversion_id, input_path, valid_formats, smoothing_level),
            daemon=True
        )
        thread.start()
        logger.info("✓ Background thread started")
        
        response_data = {
            'conversion_id': conversion_id,
            'message': 'Smooth surface conversion started',
            'input_file': filename,
            'output_formats': valid_formats,
            'smoothing_level': smoothing_level
        }
        
        logger.info(f"Returning response: {response_data}")
        return jsonify(response_data)
        
    except Exception as e:
        logger.error(f"Upload error: {str(e)}")
        logger.error(f"Full traceback: {traceback.format_exc()}")
        return jsonify({'error': f'Upload failed: {str(e)}'}), 500

def convert_file_async_debug(conversion_id, input_path, output_formats, smoothing_level='medium'):
    """Debug version of async conversion with detailed logging"""
    logger.info(f"=== STARTING ASYNC CONVERSION {conversion_id} ===")
    
    try:
        # Update progress function
        def progress_callback(message, progress=None):
            try:
                logger.info(f"[{conversion_id}] Progress: [{progress}%] {message}")
                if conversion_id in conversion_progress:
                    conversion_progress[conversion_id]['message'] = message
                    if progress is not None:
                        conversion_progress[conversion_id]['progress'] = min(int(progress), 100)
                    conversion_progress[conversion_id]['status'] = 'converting'
            except Exception as e:
                logger.error(f"Progress callback error: {e}")
        
        # Test if we can import the converter
        progress_callback("Loading PLY converter...", 5)
        
        try:
            from ply_converter import PLYConverter
            logger.info("✓ PLYConverter imported successfully")
        except Exception as import_error:
            logger.error(f"✗ PLYConverter import failed: {import_error}")
            raise RuntimeError(f"Failed to import PLYConverter: {import_error}")
        
        # Test if we can create converter instance
        progress_callback("Creating converter instance...", 10)
        
        try:
            converter = PLYConverter()
            logger.info("✓ PLYConverter instance created")
        except Exception as create_error:
            logger.error(f"✗ PLYConverter creation failed: {create_error}")
            raise RuntimeError(f"Failed to create converter: {create_error}")
        
        # Check input file
        progress_callback("Checking input file...", 15)
        
        if not os.path.exists(input_path):
            raise RuntimeError(f"Input file not found: {input_path}")
        
        file_size = os.path.getsize(input_path)
        logger.info(f"✓ Input file exists: {input_path} ({file_size} bytes)")
        
        # Check output directory
        output_dir = app.config['OUTPUT_FOLDER']
        if not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
            logger.info(f"✓ Created output directory: {output_dir}")
        
        # Start actual conversion
        progress_callback("Starting PLY conversion...", 20)
        logger.info(f"Calling converter.convert_ply with:")
        logger.info(f"  input_path: {input_path}")
        logger.info(f"  output_dir: {output_dir}")
        logger.info(f"  formats: {output_formats}")
        logger.info(f"  conversion_id: {conversion_id}")
        logger.info(f"  smoothing: {smoothing_level}")
        
        # Perform conversion
        results = converter.convert_ply(
            input_path, 
            output_dir, 
            output_formats,
            conversion_id,
            progress_callback,
            smoothing_level=smoothing_level
        )
        
        logger.info(f"✅ Conversion completed! Results: {results}")
        
        # Verify output files
        verified_results = {}
        for format_name, file_path in results.items():
            if file_path and os.path.exists(file_path):
                file_size = os.path.getsize(file_path)
                logger.info(f"✓ Output file verified: {format_name} -> {file_path} ({file_size} bytes)")
                verified_results[format_name] = file_path
            else:
                logger.warning(f"✗ Output file missing: {format_name} -> {file_path}")
        
        if not verified_results:
            raise RuntimeError("No output files were created successfully")
        
        # Store results
        conversion_results[conversion_id] = verified_results
        conversion_progress[conversion_id]['status'] = 'completed'
        conversion_progress[conversion_id]['progress'] = 100
        conversion_progress[conversion_id]['message'] = f'Conversion completed! {len(verified_results)} files created.'
        
        logger.info(f"✅ Conversion {conversion_id} completed successfully")
        
    except Exception as e:
        error_msg = f'Conversion failed: {str(e)}'
        logger.error(f"❌ Conversion {conversion_id} failed: {error_msg}")
        logger.error(f"Full traceback: {traceback.format_exc()}")
        
        # Update progress with error
        if conversion_id in conversion_progress:
            conversion_progress[conversion_id]['status'] = 'error'
            conversion_progress[conversion_id]['message'] = error_msg
    
    finally:
        # Clean up input file
        try:
            if os.path.exists(input_path):
                os.remove(input_path)
                logger.info(f"✓ Cleaned up input file: {input_path}")
        except Exception as cleanup_error:
            logger.warning(f"Input file cleanup failed: {cleanup_error}")

@app.route('/progress/<conversion_id>')
def get_progress(conversion_id):
    """Get conversion progress with detailed logging"""
    logger.info(f"Progress request for: {conversion_id}")
    
    try:
        if conversion_id not in conversion_progress:
            logger.warning(f"Progress requested for unknown conversion: {conversion_id}")
            logger.info(f"Available conversions: {list(conversion_progress.keys())}")
            return jsonify({'error': 'Conversion not found'}), 404
        
        progress_data = conversion_progress[conversion_id].copy()
        logger.info(f"Progress data: {progress_data}")
        
        # Add download links if conversion is completed
        if progress_data['status'] == 'completed' and conversion_id in conversion_results:
            results = conversion_results[conversion_id]
            progress_data['download_links'] = {}
            
            for format_name, file_path in results.items():
                if file_path and os.path.exists(file_path):
                    download_url = url_for('download_file', conversion_id=conversion_id, format_name=format_name)
                    progress_data['download_links'][format_name] = download_url
                    logger.info(f"Added download link: {format_name} -> {download_url}")
        
        logger.info(f"Returning progress: {progress_data}")
        return jsonify(progress_data)
        
    except Exception as e:
        logger.error(f"Progress check error for {conversion_id}: {str(e)}")
        logger.error(f"Full traceback: {traceback.format_exc()}")
        return jsonify({'error': f'Progress check failed: {str(e)}'}), 500

@app.route('/download/<conversion_id>/<format_name>')
def download_file(conversion_id, format_name):
    """Download converted file with logging"""
    logger.info(f"Download request: {conversion_id}/{format_name}")
    
    try:
        if conversion_id not in conversion_results:
            logger.warning(f"Download requested for unknown conversion: {conversion_id}")
            return jsonify({'error': 'Conversion not found'}), 404
        
        results = conversion_results[conversion_id]
        if format_name not in results:
            logger.warning(f"Download requested for unknown format: {format_name}")
            logger.info(f"Available formats: {list(results.keys())}")
            return jsonify({'error': 'Format not found'}), 404
        
        file_path = results[format_name]
        if not file_path or not os.path.exists(file_path):
            logger.error(f"Download file not found: {file_path}")
            return jsonify({'error': 'File not found'}), 404
        
        file_size = os.path.getsize(file_path)
        logger.info(f"Serving download: {file_path} ({file_size} bytes)")
        
        return send_file(
            file_path,
            as_attachment=True,
            download_name=f"smooth_{conversion_id}.{format_name}"
        )
        
    except Exception as e:
        logger.error(f"Download error: {str(e)}")
        return jsonify({'error': f'Download failed: {str(e)}'}), 500

@app.route('/status')
def status():
    """Status endpoint for debugging"""
    return jsonify({
        'active_conversions': len(conversion_progress),
        'completed_conversions': len(conversion_results),
        'upload_folder': app.config['UPLOAD_FOLDER'],
        'output_folder': app.config['OUTPUT_FOLDER'],
        'upload_folder_exists': os.path.exists(app.config['UPLOAD_FOLDER']),
        'output_folder_exists': os.path.exists(app.config['OUTPUT_FOLDER'])
    })

@app.route('/cleanup/<conversion_id>', methods=['POST'])
def cleanup_conversion(conversion_id):
    """Clean up conversion files and data"""
    logger.info(f"Cleanup request: {conversion_id}")
    
    try:
        # Remove from progress tracking
        if conversion_id in conversion_progress:
            del conversion_progress[conversion_id]
            logger.info("✓ Removed from progress tracking")
        
        # Remove output files and cleanup results
        if conversion_id in conversion_results:
            results = conversion_results[conversion_id]
            for file_path in results.values():
                if file_path and os.path.exists(file_path):
                    try:
                        os.remove(file_path)
                        logger.info(f"✓ Removed file: {file_path}")
                    except Exception as e:
                        logger.warning(f"Failed to remove file {file_path}: {e}")
            del conversion_results[conversion_id]
            logger.info("✓ Cleaned up results")
        
        return jsonify({'message': 'Cleanup completed'})
        
    except Exception as e:
        logger.error(f"Cleanup error: {str(e)}")
        return jsonify({'error': f'Cleanup failed: {str(e)}'}), 500

@app.errorhandler(413)
def too_large(e):
    logger.error("File too large uploaded")
    return jsonify({'error': 'File too large. Maximum size is 500MB.'}), 413

@app.errorhandler(404)
def not_found(e):
    logger.warning(f"404 error: {request.url}")
    return jsonify({'error': 'Endpoint not found'}), 404

@app.errorhandler(500)
def internal_error(e):
    logger.error(f"500 error: {str(e)}")
    return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    print("=" * 60)
    print("PLY CONVERTER DEBUG VERSION")
    print("=" * 60)
    print(f"Upload folder: {app.config['UPLOAD_FOLDER']}")
    print(f"Output folder: {app.config['OUTPUT_FOLDER']}")
    print(f"Max file size: {app.config['MAX_CONTENT_LENGTH'] // (1024*1024)}MB")
    print("")
    print("Endpoints:")
    print("  http://localhost:5000/ - Main interface")
    print("  http://localhost:5000/status - Status check")
    print("")
    print("Logs are saved to: ply_converter_app.log")
    print("=" * 60)
    
    app.run(host='0.0.0.0', port=5000, debug=True)