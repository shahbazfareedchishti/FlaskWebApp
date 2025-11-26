from flask import Flask, render_template, request, jsonify, redirect, url_for, session, flash
from models import DatabaseHelper
import json
from auth import AuthService
from sound_detector import detect_sound_from_audio
import secrets
from datetime import datetime

app = Flask(__name__)
app.secret_key = secrets.token_hex(16)

db_helper = DatabaseHelper()
auth_service = AuthService(db_helper) 
db_helper.init_db()

def is_logged_in():
    return 'user_id' in session

# Routes
@app.route('/')
def home():
    if is_logged_in():
        return redirect(url_for('main'))
    return redirect(url_for('login'))

@app.route('/login', methods=['GET', 'POST'])
def login():
    if is_logged_in():
        return redirect(url_for('main'))
    
    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        password = request.form.get('password', '').strip()

        if len(username) < 3:
            flash('Username must be at least 3 characters', 'error')
            return render_template('login.html')
        
        if len(password) < 6:
            flash('Password must be at least 6 characters', 'error')
            return render_template('login.html')
        
        if auth_service.login(username, password):
            session['user_id'] = auth_service.current_user_id
            session['username'] = auth_service.current_username
            return redirect(url_for('main'))
        else:
            flash('Invalid username or password', 'error')
    
    return render_template('login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form.get('username')
        email = request.form.get('email')
        password = request.form.get('password')
        confirm_password = request.form.get('confirm_password')
        
        if password != confirm_password:
            flash('Passwords do not match')
        else:
            try:
                auth_service.register(username, email, password)
                flash('Registration successful! Please login.')
                return redirect(url_for('login'))
            except Exception as e:
                # If request came from AJAX, return JSON error
                if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
                    return jsonify({'error': str(e)}), 400
                flash(str(e))
    
    return render_template('register.html')

@app.route('/forgot-password', methods=['GET', 'POST'])
def forgot_password():
    if request.method == 'POST':
        email = request.form.get('email')
        try:
            message = auth_service.request_password_reset(email)
            flash(message)
            return redirect(url_for('reset_password'))
        except Exception as e:
            flash(str(e))
    
    return render_template('forgot_password.html')

@app.route('/reset-password', methods=['GET', 'POST'])
def reset_password():
    if request.method == 'POST':
        token = request.form.get('token')
        new_password = request.form.get('new_password')
        confirm_password = request.form.get('confirm_password')
        
        if new_password != confirm_password:
            flash('Passwords do not match')
        else:
            if auth_service.reset_password(token, new_password):
                flash('Password reset successful! Please login.')
                return redirect(url_for('login'))
            else:
                flash('Invalid or expired reset token')
    
    return render_template('reset_password.html')

@app.route('/main')
def main():
    if not is_logged_in():
        return redirect(url_for('login'))
    return render_template('main.html')

@app.route('/snr-analysis', methods=['GET', 'POST'])
def snr_analysis():
    if not is_logged_in():
        return redirect(url_for('login'))

    # If POSTed SNR data is provided (realtime flow), use it directly
    if request.method == 'POST':
        try:
            # Accept JSON body or form-encoded 'snr_data' field
            if request.is_json:
                payload = request.get_json()
            else:
                payload_str = request.form.get('snr_data') or request.form.get('data')
                payload = json.loads(payload_str) if payload_str else {}

            # Normalize possible shapes into snr_over_time & time_bins
            snr_values = None
            time_bins = None
            total_duration = None

            # payload may be snr_metrics or already shaped for template
            if isinstance(payload, dict):
                if 'snr_values_over_time' in payload and 'time_bins' in payload:
                    snr_values = payload.get('snr_values_over_time')
                    time_bins = payload.get('time_bins')
                    total_duration = payload.get('total_duration')
                elif 'snr_over_time' in payload and 'time_bins' in payload:
                    snr_values = payload.get('snr_over_time')
                    time_bins = payload.get('time_bins')
                    total_duration = payload.get('total_duration')
                else:
                    # Try nested 'snr_metrics'
                    sm = payload.get('snr_metrics') or payload.get('snr_analysis')
                    if isinstance(sm, dict):
                        snr_values = sm.get('snr_values_over_time') or sm.get('snr_over_time')
                        time_bins = sm.get('time_bins')
                        total_duration = sm.get('total_duration')

            if not snr_values or not time_bins:
                flash('No SNR time-series provided', 'error')
                return redirect(url_for('main'))

            snr_data = {
                'snr_over_time': snr_values,
                'time_bins': time_bins,
                'total_duration': total_duration
            }

            return render_template('snr_analysis.html', snr_data=snr_data)

        except Exception as e:
            flash('Error reading SNR data: ' + str(e), 'error')
            return redirect(url_for('main'))

    # GET fallback: try to read latest stored detection (if available)
    detections = db_helper.get_user_detections(session['user_id'])
    if not detections:
        flash('No detection data available', 'error')
        return redirect(url_for('main'))

    latest_detection = detections[0]
    try:
        raw_json = latest_detection.get('raw_data')
        if not raw_json:
            flash('No SNR data stored for the latest detection', 'error')
            return redirect(url_for('main'))

        raw_data = json.loads(raw_json)
        snr_metrics = raw_data.get('snr_metrics', {})
        snr_values = None
        time_bins = None

        if 'snr_values_over_time' in snr_metrics and 'time_bins' in snr_metrics:
            snr_values = snr_metrics.get('snr_values_over_time')
            time_bins = snr_metrics.get('time_bins')
        elif 'snr_over_time' in raw_data and 'time_bins' in raw_data:
            snr_values = raw_data.get('snr_over_time')
            time_bins = raw_data.get('time_bins')

        if not snr_values or not time_bins:
            flash('No SNR time-series data available', 'error')
            return redirect(url_for('main'))

        snr_data = {
            'snr_over_time': snr_values,
            'time_bins': time_bins,
            'total_duration': snr_metrics.get('total_duration') if isinstance(snr_metrics, dict) else raw_data.get('total_duration')
        }

    except (json.JSONDecodeError, KeyError):
        flash('Error parsing SNR data', 'error')
        return redirect(url_for('main'))

    return render_template('snr_analysis.html', snr_data=snr_data)

# Update the detect route to include SNR data in response
@app.route('/detect', methods=['POST'])
def detect():
    if not is_logged_in():
        return jsonify({'success': False, 'error': 'Not authenticated'})
    
    if 'audio' not in request.files:
        return jsonify({'success': False, 'error': 'No audio file'})
    
    audio_file = request.files['audio']
    result = detect_sound_from_audio(audio_file)

    # Ensure client-friendly SNR key exists
    if 'snr_analysis' in result and 'snr_metrics' not in result:
        result['snr_metrics'] = result.get('snr_analysis')

    # If server produced a spectrogram image, provide a static URL for the frontend
    try:
        if result.get('spectrogram_plot'):
            import os
            filename = os.path.basename(result.get('spectrogram_plot'))
            # url_for is already imported
            result['spectrogram_url'] = url_for('static', filename=f'plots/{filename}')
    except Exception:
        pass

    if result.get('success'):
        # Save detection and persist raw analysis JSON for later review
        raw_data = {
            'predicted_class': result.get('predicted_class'),
            'confidence': result.get('confidence'),
            'snr_metrics': result.get('snr_metrics') or result.get('snr_analysis') or {},
            'all_predictions': result.get('all_predictions', {})
        }
        try:
            db_helper.insert_detection_with_raw(
                user_id=session['user_id'],
                sound_class=result.get('predicted_class'),
                confidence=result.get('confidence'),
                raw_data=json.dumps(raw_data)
            )
        except Exception as e:
            print(f"Error inserting detection with raw data: {e}")
            # Fallback to minimal insert if raw insert fails
            db_helper.insert_detection(
                user_id=session['user_id'],
                sound_class=result.get('predicted_class'),
                confidence=result.get('confidence')
            )

    return jsonify(result)

@app.template_filter('format_datetime')
def format_datetime(value):
    """Format datetime for display"""
    if not value:
        return "N/A"

    if hasattr(value, 'strftime'):
        return value.strftime('%Y-%m-%d')
    
    if isinstance(value, str):
        try:
            value = datetime.fromisoformat(value.replace('Z', '+00:00'))
            return value.strftime('%Y-%m-%d')
        except ValueError:
            try:
                value = datetime.strptime(value, '%Y-%m-%d %H:%M:%S')
                return value.strftime('%Y-%m-%d')
            except ValueError:
                return value
    return str(value)

@app.template_filter('get_sound_icon')
def get_sound_icon(sound_class):
    """Get appropriate icon for sound class"""
    icons = {
        'SpeedBoat': 'fas fa-ship',
        'UUV': 'fas fa-satellite',
        'KaiYuan': 'fas fa-anchor',
        'Noise': 'fas fa-volume-mute',
        'Unknown': 'fas fa-question-circle'
    }
    return icons.get(sound_class, 'fas fa-question-circle')

@app.route('/logs')
def logs():
    if not is_logged_in():
        return redirect(url_for('login'))
    
    detections = db_helper.get_user_detections(session['user_id'])
    return render_template('logs.html', detections=detections)

@app.route('/manage-account')
def manage_account():
    if not is_logged_in():
        return redirect(url_for('login'))
    
    user = db_helper.get_user_by_id(session['user_id'])
    # Convert created_at ISO string to datetime for template usage
    if user and isinstance(user.get('created_at'), str):
        try:
            user['created_at'] = datetime.fromisoformat(user['created_at'])
        except Exception:
            user['created_at'] = None
    
    # Get user's detection count for stats
    detections = db_helper.get_user_detections(session['user_id'])
    detections_count = len(detections)
    
    return render_template('manage_account.html', 
                         user=user, 
                         detections_count=detections_count)

@app.route('/update-account', methods=['POST'])
def update_account():
    if not is_logged_in():
        return redirect(url_for('login'))
    
    username = request.form.get('username')
    email = request.form.get('email')
    current_password = request.form.get('current_password')
    new_password = request.form.get('new_password')
    confirm_password = request.form.get('confirm_password')
    
    # Verify current password first
    user = db_helper.get_user_by_id(session['user_id'])
    if not db_helper.login_user(user['username'], current_password):
        flash('Current password is incorrect', 'error')
        return redirect(url_for('manage_account'))
    
    # Update username if changed
    if username and username != user['username']:
        try:
            # Check if username already exists
            existing_user = db_helper.get_user_by_email(username)  # Using email check for username availability
            if existing_user and existing_user['id'] != session['user_id']:
                flash('Username already exists', 'error')
            else:
                flash('Username update functionality not implemented yet', 'warning')
        except Exception as e:
            flash(str(e), 'error')
    
    # Update email if changed
    if email and email != user['email']:
        try:
            # Check if email already exists
            existing_user = db_helper.get_user_by_email(email)
            if existing_user and existing_user['id'] != session['user_id']:
                flash('Email already exists', 'error')
            else:
                flash('Email update functionality not implemented yet', 'warning')
        except Exception as e:
            flash(str(e), 'error')
    
    # Update password if provided
    if new_password:
        if new_password != confirm_password:
            flash('New passwords do not match', 'error')
        elif len(new_password) < 6:
            flash('New password must be at least 6 characters', 'error')
        else:
            try:
                db_helper.update_user_password(user['email'], new_password)
                flash('Password updated successfully', 'success')
            except Exception as e:
                flash(str(e), 'error')
    
    return redirect(url_for('manage_account'))

@app.route('/delete-account', methods=['POST'])
def delete_account():
    if not is_logged_in():
        return redirect(url_for('login'))
    
    user_id = session['user_id']
    current_password = request.form.get('current_password', '').strip()
    
    # Verify current password for security
    if not current_password:
        flash('Please enter your current password to delete your account', 'error')
        return redirect(url_for('manage_account'))
    
    user = db_helper.get_user_by_id(user_id)
    if not user:
        flash('User not found', 'error')
        return redirect(url_for('manage_account'))
    
    if not db_helper.login_user(user['username'], current_password):
        flash('Current password is incorrect', 'error')
        return redirect(url_for('manage_account'))
    
    try:
        # Clear user's detections first
        db_helper.clear_user_detections(user_id)
        
        # Delete user account
        db_helper.delete_user(user_id)
        
        # Logout and clear session
        auth_service.logout()
        session.clear()
        
        flash('Your account has been successfully deleted', 'success')
        return redirect(url_for('login'))
        
    except Exception as e:
        flash(f'Error deleting account: {str(e)}', 'error')
        return redirect(url_for('manage_account'))

@app.route('/logout')
def logout():
    auth_service.logout()
    session.clear()
    return redirect(url_for('login'))

if __name__ == '__main__':
    app.run(debug=True, port=5000)