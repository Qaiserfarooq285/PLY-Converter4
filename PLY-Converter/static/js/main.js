class PLYSmoothConverter {
    constructor() {
        this.uploadArea = document.getElementById('uploadArea');
        this.fileInput = document.getElementById('fileInput');
        this.browseBtn = document.getElementById('browseBtn');
        this.convertBtn = document.getElementById('convertBtn');
        this.progressCard = document.getElementById('progressCard');
        this.resultsCard = document.getElementById('resultsCard');
        this.errorAlert = document.getElementById('errorAlert');
        this.newConversionBtn = document.getElementById('newConversionBtn');
        
        this.currentFile = null;
        this.conversionId = null;
        this.progressInterval = null;
        this.progressRetries = 0;
        this.maxRetries = 3;
        
        this.initializeEventListeners();
    }
    
    initializeEventListeners() {
        // File input events
        this.fileInput.addEventListener('change', (e) => this.handleFileSelect(e));
        this.browseBtn.addEventListener('click', () => this.fileInput.click());
        this.convertBtn.addEventListener('click', () => this.startConversion());
        this.newConversionBtn.addEventListener('click', () => this.resetForm());
        
        // Drag and drop events
        this.uploadArea.addEventListener('dragover', (e) => this.handleDragOver(e));
        this.uploadArea.addEventListener('dragleave', (e) => this.handleDragLeave(e));
        this.uploadArea.addEventListener('drop', (e) => this.handleDrop(e));
        this.uploadArea.addEventListener('click', () => this.fileInput.click());
        
        // Format checkbox events
        const formatCheckboxes = document.querySelectorAll('input[type="checkbox"][value]');
        formatCheckboxes.forEach(checkbox => {
            checkbox.addEventListener('change', () => this.updateConvertButton());
        });
        
        // Smoothing radio button events
        const smoothingRadios = document.querySelectorAll('input[name="smoothing"]');
        smoothingRadios.forEach(radio => {
            radio.addEventListener('change', () => this.updateConvertButton());
        });
    }
    
    handleDragOver(e) {
        e.preventDefault();
        this.uploadArea.classList.add('dragover');
    }
    
    handleDragLeave(e) {
        e.preventDefault();
        this.uploadArea.classList.remove('dragover');
    }
    
    handleDrop(e) {
        e.preventDefault();
        this.uploadArea.classList.remove('dragover');
        
        const files = e.dataTransfer.files;
        if (files.length > 0) {
            this.setSelectedFile(files[0]);
        }
    }
    
    handleFileSelect(e) {
        if (e.target.files.length > 0) {
            this.setSelectedFile(e.target.files[0]);
        }
    }
    
    setSelectedFile(file) {
        // Validate file type
        if (!file.name.toLowerCase().endsWith('.ply')) {
            this.showError('Please select a PLY file (.ply extension)');
            return;
        }
        
        // Check file size (500MB limit)
        const maxSize = 500 * 1024 * 1024;
        if (file.size > maxSize) {
            this.showError('File size too large. Maximum size is 500MB.');
            return;
        }
        
        this.currentFile = file;
        this.updateUploadArea();
        this.updateConvertButton();
        this.hideError();
    }
    
    updateUploadArea() {
        if (this.currentFile) {
            this.uploadArea.classList.add('file-selected');
            this.uploadArea.innerHTML = `
                <div class="upload-content">
                    <i class="fas fa-file-alt upload-icon mb-3"></i>
                    <h5>File Selected for Smoothing</h5>
                    <p class="text-muted mb-2">${this.currentFile.name}</p>
                    <p class="small text-muted">${this.formatFileSize(this.currentFile.size)}</p>
                    <button type="button" class="btn btn-outline-secondary btn-sm mt-2">
                        <i class="fas fa-times me-1"></i>Change File
                    </button>
                </div>
            `;
        }
    }
    
    updateConvertButton() {
        const hasFile = this.currentFile !== null;
        const hasFormats = this.getSelectedFormats().length > 0;
        const smoothingLevel = this.getSelectedSmoothingLevel();
        
        this.convertBtn.disabled = !hasFile || !hasFormats;
        
        if (!hasFormats && hasFile) {
            this.convertBtn.innerHTML = '<i class="fas fa-exclamation-triangle me-2"></i>Select Output Format(s)';
        } else if (hasFile && hasFormats) {
            const smoothingText = this.getSmoothingDisplayText(smoothingLevel);
            this.convertBtn.innerHTML = `<i class="fas fa-magic me-2"></i>Create ${smoothingText} Surface`;
        } else {
            this.convertBtn.innerHTML = '<i class="fas fa-magic me-2"></i>Create Smooth Surface';
        }
    }
    
    getSelectedFormats() {
        const checkboxes = document.querySelectorAll('input[type="checkbox"][value]:checked');
        return Array.from(checkboxes).map(cb => cb.value);
    }
    
    getSelectedSmoothingLevel() {
        const selectedRadio = document.querySelector('input[name="smoothing"]:checked');
        return selectedRadio ? selectedRadio.value : 'medium';
    }
    
    getSmoothingDisplayText(level) {
        const displayTexts = {
            'light': 'Lightly Smoothed',
            'medium': 'Smoothed',
            'high': 'Highly Smoothed',
            'ultra': 'Ultra-Smooth'
        };
        return displayTexts[level] || 'Smoothed';
    }
    
    async startConversion() {
        if (!this.currentFile) {
            this.showError('Please select a PLY file first');
            return;
        }
        
        const formats = this.getSelectedFormats();
        if (formats.length === 0) {
            this.showError('Please select at least one output format');
            return;
        }
        
        const smoothingLevel = this.getSelectedSmoothingLevel();
        
        this.hideError();
        this.showProgress();
        this.progressRetries = 0;
        
        try {
            // Prepare form data
            const formData = new FormData();
            formData.append('file', this.currentFile);
            formData.append('smoothing', smoothingLevel);
            formats.forEach(format => formData.append('formats', format));
            
            console.log('Starting upload...', {
                fileName: this.currentFile.name,
                fileSize: this.currentFile.size,
                smoothing: smoothingLevel,
                formats: formats
            });
            
            // Upload and start conversion
            const response = await fetch('/upload', {
                method: 'POST',
                body: formData
            });
            
            const responseText = await response.text();
            console.log('Upload response:', responseText);
            
            if (!response.ok) {
                let error;
                try {
                    error = JSON.parse(responseText);
                } catch (e) {
                    error = { error: `Server error: ${response.status}` };
                }
                throw new Error(error.error || 'Upload failed');
            }
            
            const result = JSON.parse(responseText);
            this.conversionId = result.conversion_id;
            
            console.log('Conversion started:', result);
            
            // Update file info
            const smoothingDisplayText = this.getSmoothingDisplayText(result.smoothing_level);
            document.getElementById('fileInfo').innerHTML = `
                <strong>File:</strong> ${result.input_file}<br>
                <strong>Smoothing Level:</strong> ${smoothingDisplayText} (${result.smoothing_level})<br>
                <strong>Output Formats:</strong> ${result.output_formats.join(', ').toUpperCase()}<br>
                <strong>Conversion ID:</strong> ${result.conversion_id}
            `;
            
            // Start progress polling
            this.startProgressPolling();
            
        } catch (error) {
            console.error('Conversion start error:', error);
            this.hideProgress();
            this.showError(`Conversion failed: ${error.message}`);
        }
    }
    
    startProgressPolling() {
        console.log('Starting progress polling for:', this.conversionId);
        this.progressInterval = setInterval(() => {
            this.updateProgress();
        }, 2000); // Poll every 2 seconds
    }
    
    async updateProgress() {
        if (!this.conversionId) {
            console.warn('No conversion ID for progress update');
            return;
        }
        
        try {
            console.log('Checking progress for:', this.conversionId);
            
            const response = await fetch(`/progress/${this.conversionId}`, {
                method: 'GET',
                headers: {
                    'Cache-Control': 'no-cache'
                }
            });
            
            if (!response.ok) {
                console.error('Progress response not OK:', response.status, response.statusText);
                throw new Error(`Progress check failed: ${response.status} ${response.statusText}`);
            }
            
            const progress = await response.json();
            console.log('Progress update:', progress);
            
            // Reset retry counter on successful fetch
            this.progressRetries = 0;
            
            // Update progress bar
            const progressBar = document.getElementById('progressBar');
            const progressPercent = document.getElementById('progressPercent');
            const progressMessage = document.getElementById('progressMessage');
            
            const progressValue = Math.max(0, Math.min(100, progress.progress || 0));
            
            progressBar.style.width = `${progressValue}%`;
            progressBar.setAttribute('aria-valuenow', progressValue);
            progressPercent.textContent = `${progressValue}%`;
            progressMessage.textContent = progress.message || 'Processing smooth surface...';
            
            // Update progress bar color based on stage
            progressBar.className = 'progress-bar progress-bar-striped progress-bar-animated';
            if (progressValue >= 80) {
                progressBar.classList.add('bg-success');
            } else if (progressValue >= 40) {
                progressBar.classList.add('bg-warning');
            } else {
                progressBar.classList.add('bg-info');
            }
            
            // Check if completed
            if (progress.status === 'completed') {
                console.log('Conversion completed!');
                clearInterval(this.progressInterval);
                this.showResults(progress);
            } else if (progress.status === 'error') {
                console.error('Conversion failed:', progress.message);
                clearInterval(this.progressInterval);
                this.hideProgress();
                this.showError(progress.message || 'Conversion failed');
            }
            
        } catch (error) {
            console.error('Progress update error:', error);
            this.progressRetries++;
            
            if (this.progressRetries >= this.maxRetries) {
                console.error('Max retries reached, stopping progress polling');
                clearInterval(this.progressInterval);
                this.hideProgress();
                this.showError(`Progress update failed after ${this.maxRetries} attempts: ${error.message}`);
            } else {
                console.log(`Progress retry ${this.progressRetries}/${this.maxRetries}`);
            }
        }
    }
    
    showResults(progress) {
        this.hideProgress();
        
        const downloadLinks = document.getElementById('downloadLinks');
        downloadLinks.innerHTML = '';
        
        if (progress.download_links) {
            const formatIcons = {
                'stl': 'fas fa-cube',
                'obj': 'fas fa-shapes',
                'glb': 'fas fa-globe',
                '3mf': 'fas fa-print',
                'dxf': 'fas fa-drafting-compass'
            };
            
            const formatDescriptions = {
                'stl': '3D Printing Ready',
                'obj': 'With Colors & Textures',
                'glb': 'Web & AR Compatible',
                '3mf': 'Advanced 3D Printing',
                'dxf': 'CAD Compatible'
            };
            
            Object.entries(progress.download_links).forEach(([format, url]) => {
                const link = document.createElement('a');
                link.href = url;
                link.className = 'download-link';
                const icon = formatIcons[format] || 'fas fa-download';
                const description = formatDescriptions[format] || 'Download File';
                
                link.innerHTML = `
                    <div class="d-flex align-items-center">
                        <i class="${icon} me-3" style="font-size: 1.2rem;"></i>
                        <div class="flex-grow-1">
                            <strong>Download Smooth ${format.toUpperCase()} File</strong>
                            <div class="small text-light opacity-75">${description}</div>
                        </div>
                        <i class="fas fa-download ms-2"></i>
                    </div>
                `;
                downloadLinks.appendChild(link);
            });
        } else {
            downloadLinks.innerHTML = '<p class="text-muted">No download links available</p>';
        }
        
        this.resultsCard.style.display = 'block';
        this.resultsCard.scrollIntoView({ behavior: 'smooth' });
    }
    
    showProgress() {
        this.progressCard.style.display = 'block';
        this.resultsCard.style.display = 'none';
        
        // Reset progress
        const progressBar = document.getElementById('progressBar');
        const progressPercent = document.getElementById('progressPercent');
        const progressMessage = document.getElementById('progressMessage');
        
        progressBar.style.width = '0%';
        progressBar.setAttribute('aria-valuenow', 0);
        progressBar.className = 'progress-bar progress-bar-striped progress-bar-animated bg-info';
        progressPercent.textContent = '0%';
        progressMessage.textContent = 'Initializing smooth surface reconstruction...';
        
        this.progressCard.scrollIntoView({ behavior: 'smooth' });
    }
    
    hideProgress() {
        this.progressCard.style.display = 'none';
        if (this.progressInterval) {
            clearInterval(this.progressInterval);
            this.progressInterval = null;
        }
    }
    
    showError(message) {
        console.error('Showing error:', message);
        this.errorAlert.style.display = 'block';
        document.getElementById('errorMessage').textContent = message;
        this.errorAlert.scrollIntoView({ behavior: 'smooth' });
    }
    
    hideError() {
        this.errorAlert.style.display = 'none';
    }
    
    async resetForm() {
        // Cleanup previous conversion
        if (this.conversionId) {
            try {
                console.log('Cleaning up conversion:', this.conversionId);
                await fetch(`/cleanup/${this.conversionId}`, { method: 'POST' });
            } catch (error) {
                console.warn('Cleanup failed:', error);
            }
        }
        
        // Reset form state
        this.currentFile = null;
        this.conversionId = null;
        this.progressRetries = 0;
        
        // Reset UI
        this.uploadArea.classList.remove('file-selected');
        this.uploadArea.innerHTML = `
            <div class="upload-content">
                <i class="fas fa-cloud-upload-alt upload-icon mb-3"></i>
                <h5>Drag & Drop PLY File Here</h5>
                <p class="text-muted mb-3">or click to browse files</p>
                <button type="button" class="btn btn-outline-primary" id="browseBtn">
                    <i class="fas fa-folder-open me-2"></i>Browse Files
                </button>
            </div>
        `;
        
        // Re-attach event listener for new browse button
        document.getElementById('browseBtn').addEventListener('click', () => this.fileInput.click());
        
        this.fileInput.value = '';
        this.hideProgress();
        this.hideError();
        this.resultsCard.style.display = 'none';
        
        // Reset smoothing level to medium
        document.getElementById('smoothMedium').checked = true;
        
        this.updateConvertButton();
        
        // Scroll to top
        window.scrollTo({ top: 0, behavior: 'smooth' });
    }
    
    formatFileSize(bytes) {
        if (bytes === 0) return '0 Bytes';
        
        const k = 1024;
        const sizes = ['Bytes', 'KB', 'MB', 'GB'];
        const i = Math.floor(Math.log(bytes) / Math.log(k));
        
        return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
    }
}

// Initialize the converter when the page loads
document.addEventListener('DOMContentLoaded', () => {
    console.log('Initializing PLY Smooth Converter...');
    new PLYSmoothConverter();
    
    // Add some interactive elements
    
    // Smoothing level tooltips and animations
    const smoothingRadios = document.querySelectorAll('input[name="smoothing"]');
    smoothingRadios.forEach(radio => {
        radio.addEventListener('change', function() {
            // Add visual feedback when smoothing level changes
            const labels = document.querySelectorAll('.form-check-label[for^="smooth"]');
            labels.forEach(label => label.classList.remove('text-primary', 'fw-bold'));
            
            const selectedLabel = document.querySelector(`label[for="${this.id}"]`);
            if (selectedLabel) {
                selectedLabel.classList.add('text-primary', 'fw-bold');
                
                // Brief animation
                selectedLabel.style.transform = 'scale(1.05)';
                setTimeout(() => {
                    selectedLabel.style.transform = 'scale(1)';
                }, 150);
            }
        });
    });
    
    // Format checkbox animations
    const formatCheckboxes = document.querySelectorAll('input[type="checkbox"][value]');
    formatCheckboxes.forEach(checkbox => {
        checkbox.addEventListener('change', function() {
            const label = this.closest('.form-check');
            if (this.checked) {
                label.style.background = 'rgba(0, 123, 255, 0.1)';
                label.style.transform = 'scale(1.02)';
            } else {
                label.style.background = 'rgba(0, 123, 255, 0.05)';
                label.style.transform = 'scale(1)';
            }
        });
    });
    
    // Add smooth scroll behavior for all internal links
    document.querySelectorAll('a[href^="#"]').forEach(anchor => {
        anchor.addEventListener('click', function (e) {
            e.preventDefault();
            const target = document.querySelector(this.getAttribute('href'));
            if (target) {
                target.scrollIntoView({
                    behavior: 'smooth'
                });
            }
        });
    });
    
    console.log('PLY Smooth Converter initialized successfully');
});