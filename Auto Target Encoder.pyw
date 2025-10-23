from __future__ import annotations 

from sklearn .linear_model import LinearRegression 
from sklearn .ensemble import RandomForestRegressor 
import customtkinter as ctk 
from customtkinter import filedialog 
import tkinter as tk 
from tkinter import ttk ,messagebox 
import threading 
import logging 
import platform 
import queue 
import time 
import json 
import uuid 
import os 
import sys 
import gc 
import subprocess 
import configparser 
from pathlib import Path 
from datetime import datetime 
from typing import Dict ,List ,Optional ,Tuple 
import concurrent .futures 
import numpy as np 
import psutil 
import itertools 
import hashlib 
import sqlite3 
import tempfile 
import re 
from dataclasses import dataclass 
import select 
import pickle 
import warnings 
warnings .filterwarnings ('ignore')


ctk .set_appearance_mode ("dark")
ctk .set_default_color_theme ("blue")

SUBPROCESS_FLAGS =0 
if os .name =='nt':
    SUBPROCESS_FLAGS =subprocess .CREATE_NO_WINDOW 


def log_to_worker (task_id ,message ):
    """Default log function for workers"""
    print (f"[Worker {task_id }] {message }")


@dataclass 
class EncodingSettings :
    ffmpeg_path :str 
    ffprobe_path :str 
    database_path :str 
    encoding_log_path :str 
    ffvship_path :str 

    max_workers :int 
    num_parallel_vmaf_runs :int 
    max_iterations :int 
    cpu_threads :int 

    encoder_type :str 
    nvenc_preset :str 
    nvenc_quality_mode :str 
    nvenc_advanced_params :str 
    svt_av1_preset :int 
    svt_av1_advanced_params :str 

    quality_metric_mode :str 
    target_score :float 
    quality_tolerance_percent :float 
    cq_search_min :int 
    cq_search_max :int 
    vmaf_targeting_mode :str 
    vmaf_target_percentile :float 

    sampling_method :str 
    sample_segment_duration :int 
    num_samples :int 
    master_sample_encoder :str 
    min_scene_changes_required :int 
    min_keyframes_required :int 
    skip_start_seconds :int 
    skip_end_seconds :int 
    ffmpeg_scenedetect_threshold :float 

    min_duration_seconds :int 
    min_filesize_mb :int 
    min_bitrate_4k_kbps :int 
    min_bitrate_1080p_kbps :int 
    min_bitrate_720p_kbps :int 

    enable_quality_cache :bool 
    enable_performance_log :bool 

    delete_source_file :bool 
    output_suffix :str 
    output_directory :str 
    use_different_input_directory :bool 
    input_directory :str 
    min_size_reduction_threshold :float 
    rename_skipped_files :bool 
    skipped_file_suffix :str
    skipped_file_filter_suffix: str 
    skip_encoding_if_target_not_reached :bool 

    output_bit_depth :str 
    ml_extra_cq_check :bool 


class FeatureExtractor :
    """Centralized feature extraction for consistent ML features across the application."""

    @staticmethod 
    def extract_video_features (media_info :dict ,complexity_data :dict =None )->dict :
        """Extract all ML-relevant features from media info and complexity data."""
        features ={
        'resolution_pixels':0 ,
        'width':0 ,
        'height':0 ,
        'source_bitrate_kbps':0 ,
        'bitrate_per_pixel':0 ,
        'complexity_score':0.5 ,
        'scenes_per_minute':0 ,
        'frame_rate':30.0 ,
        'duration_seconds':0 ,
        'aspect_ratio':1.778 ,
        'is_hdr':0 ,
        'is_10bit':0 
        }

        if not media_info :
            return features 


        if 'format'in media_info :
            format_info =media_info ['format']
            features ['duration_seconds']=float (format_info .get ('duration',0 ))

            if features ['duration_seconds']>0 :
                file_size_bits =float (format_info .get ('bit_rate',0 ))
                if file_size_bits ==0 and 'size'in format_info :
                    file_size_bits =float (format_info .get ('size',0 ))*8 
                features ['source_bitrate_kbps']=file_size_bits /1000 


        video_stream =next ((s for s in media_info .get ('streams',[])
        if s .get ('codec_type')=='video'),None )

        if video_stream :
            features ['width']=video_stream .get ('width',0 )
            features ['height']=video_stream .get ('height',0 )
            features ['resolution_pixels']=features ['width']*features ['height']


            if features ['height']>0 :
                features ['aspect_ratio']=features ['width']/features ['height']


            try :
                frame_rate_str =video_stream .get ('avg_frame_rate','30/1')
                if '/'in frame_rate_str :
                    num ,den =map (float ,frame_rate_str .split ('/'))
                    if den >0 :
                        features ['frame_rate']=num /den 
                else :
                    features ['frame_rate']=float (frame_rate_str )
            except :
                features ['frame_rate']=30.0 


            if features ['resolution_pixels']>0 and features ['source_bitrate_kbps']>0 :
                features ['bitrate_per_pixel']=(features ['source_bitrate_kbps']*1000 )/features ['resolution_pixels']


            pix_fmt =video_stream .get ('pix_fmt','')
            if '10'in pix_fmt or '12'in pix_fmt :
                features ['is_10bit']=1 

            color_transfer =video_stream .get ('color_transfer','')
            if 'smpte2084'in color_transfer or 'arib-std-b67'in color_transfer :
                features ['is_hdr']=1 


        if complexity_data :
            features ['complexity_score']=complexity_data .get ('complexity_score',0.5 )
            features ['scenes_per_minute']=complexity_data .get ('scenes_per_minute',0 )
            features ['scene_count']=complexity_data .get ('scene_count',0 )
            features ['avg_scene_duration']=complexity_data .get ('avg_scene_duration',0 )

        return features 

    @staticmethod 
    def get_encoder_features (settings :EncodingSettings )->dict :
        """Extract encoder-specific features."""
        features ={
        'is_nvenc':1 if settings .encoder_type =='nvenc'else 0 ,
        'preset_num':0 ,
        'is_10bit_output':0 
        }

        if settings .encoder_type =='nvenc':

            preset_str =settings .nvenc_preset .lower ()
            if preset_str .startswith ('p'):
                try :
                    features ['preset_num']=int (preset_str [1 :])
                except :
                    features ['preset_num']=5 
            else :
                features ['preset_num']=5 
        else :
            features ['preset_num']=settings .svt_av1_preset 

        if settings .output_bit_depth =='10bit':
            features ['is_10bit_output']=1 

        return features 






class PredictionErrorAnalyzer:
    """Analyzes and learns from ML prediction errors to improve future predictions."""
    
    def __init__(self, database_manager=None):
        self.database_manager = database_manager
        self.error_patterns = {}
        self.content_signatures = {}
        
    def get_content_signature(self, features: dict) -> str:
        """Create a signature for content type based on key features."""
        # Group by resolution, complexity, and bitrate ranges
        resolution_bucket = "4k" if features.get('resolution_pixels', 0) > 3000000 else \
                           "1080p" if features.get('resolution_pixels', 0) > 1000000 else \
                           "720p"
        
        complexity_bucket = "high" if features.get('complexity_score', 0.5) > 0.7 else \
                           "medium" if features.get('complexity_score', 0.5) > 0.3 else \
                           "low"
        
        motion_bucket = "high" if features.get('scenes_per_minute', 0) > 30 else \
                       "medium" if features.get('scenes_per_minute', 0) > 10 else \
                       "low"
        
        return f"{resolution_bucket}_{complexity_bucket}_{motion_bucket}"
    
    def analyze_error(self, features: dict, predicted_cq: int, optimal_cq: int, 
                     predicted_score: float, actual_score: float) -> float:
        """Analyze prediction error and return correction factor."""
        signature = self.get_content_signature(features)
        
        if signature not in self.error_patterns:
            self.error_patterns[signature] = []
        
        error_data = {
            'predicted_cq': predicted_cq,
            'optimal_cq': optimal_cq,
            'cq_error': optimal_cq - predicted_cq,
            'score_error': actual_score - predicted_score,
            'timestamp': time.time()
        }
        
        self.error_patterns[signature].append(error_data)
        
        # Keep only recent errors (last 20 per signature)
        self.error_patterns[signature] = self.error_patterns[signature][-20:]
        
        # Calculate average correction needed for this content type
        if len(self.error_patterns[signature]) >= 3:
            recent_errors = self.error_patterns[signature][-10:]
            avg_cq_error = np.mean([e['cq_error'] for e in recent_errors])
            return avg_cq_error
        
        return 0.0
    
    def get_correction_factor(self, features: dict) -> float:
        """Get learned correction factor for this content type."""
        signature = self.get_content_signature(features)
        
        if signature in self.error_patterns and len(self.error_patterns[signature]) >= 3:
            recent_errors = self.error_patterns[signature][-10:]
            return np.mean([e['cq_error'] for e in recent_errors])
        
        return 0.0




class PerformanceErrorAnalyzer:
    """Real-time ETA correction based on recent prediction errors."""
    
    def __init__(self):
        self.error_history = {}  # Keyed by content signature
        self.correction_factors = {}
        
    def get_content_signature(self, features: dict, encoder_features: dict) -> str:
        """Create signature for similar encoding scenarios."""
        resolution = "4k" if features.get('resolution_pixels', 0) > 3000000 else \
                    "1080p" if features.get('resolution_pixels', 0) > 1000000 else "720p"
        
        complexity = "high" if features.get('complexity_score', 0.5) > 0.7 else \
                    "medium" if features.get('complexity_score', 0.5) > 0.3 else "low"
        
        encoder = "nvenc" if encoder_features.get('is_nvenc') else "svt"
        preset = encoder_features.get('preset_num', 5)
        
        return f"{encoder}_{preset}_{resolution}_{complexity}"
    
    def record_error(self, features: dict, encoder_features: dict, 
                     predicted_fps: float, actual_fps: float):
        """Record prediction error for learning."""
        signature = self.get_content_signature(features, encoder_features)
        
        if signature not in self.error_history:
            self.error_history[signature] = []
        
        error_ratio = actual_fps / predicted_fps if predicted_fps > 0 else 1.0
        self.error_history[signature].append({
            'ratio': error_ratio,
            'timestamp': time.time()
        })
        
        # Keep only recent errors (last 10)
        self.error_history[signature] = self.error_history[signature][-10:]
        
        # Update correction factor if we have enough data
        if len(self.error_history[signature]) >= 3:
            recent_ratios = [e['ratio'] for e in self.error_history[signature][-5:]]
            self.correction_factors[signature] = np.median(recent_ratios)
    
    def get_corrected_fps(self, predicted_fps: float, features: dict, 
                         encoder_features: dict) -> float:
        """Apply learned correction to FPS prediction."""
        signature = self.get_content_signature(features, encoder_features)
        
        if signature in self.correction_factors:
            correction = self.correction_factors[signature]
            # Apply correction with dampening to avoid overcorrection
            dampened_correction = 1.0 + (correction - 1.0) * 0.7
            corrected = predicted_fps * dampened_correction
            return max(corrected, 1.0)  # Ensure positive FPS
        
        return predicted_fps






def get_vmaf_subtype(settings: EncodingSettings) -> str:
    """Get the VMAF subtype string based on current settings."""
    if settings.quality_metric_mode != 'vmaf':
        return None
    
    if settings.vmaf_targeting_mode == 'average':
        return 'average'
    elif settings.vmaf_targeting_mode == 'percentile':
        return f'percentile_{int(settings.vmaf_target_percentile)}'
    else:
        return 'average'  # Default fallback





class ModelPersistence :
    """Handles saving and loading of trained models."""

    def __init__ (self ,model_dir :str =None ):
        if model_dir is None :
            model_dir =os .path .join (os .path .dirname (os .path .abspath (__file__ )),'ml_models')
        self .model_dir =Path (model_dir )
        self .model_dir .mkdir (exist_ok =True )

    def save_model (self ,model ,model_name :str ,metadata :dict =None ):
        """Save a trained model with metadata."""
        model_path =self .model_dir /f"{model_name }.pkl"
        meta_path =self .model_dir /f"{model_name }_meta.json"

        try :

            with open (model_path ,'wb')as f :
                pickle .dump (model ,f )


            if metadata is None :
                metadata ={}
            metadata ['saved_at']=time .time ()
            metadata ['model_name']=model_name 

            with open (meta_path ,'w')as f :
                json .dump (metadata ,f ,indent =2 )

            return True 
        except Exception as e :
            print (f"Error saving model {model_name }: {e }")
            return False 

    def load_model (self ,model_name :str )->tuple :
        """Load a model and its metadata."""
        model_path =self .model_dir /f"{model_name }.pkl"
        meta_path =self .model_dir /f"{model_name }_meta.json"

        if not model_path .exists ():
            return None ,None 

        try :

            with open (model_path ,'rb')as f :
                model =pickle .load (f )


            metadata ={}
            if meta_path .exists ():
                with open (meta_path ,'r')as f :
                    metadata =json .load (f )

            return model ,metadata 
        except Exception as e :
            print (f"Error loading model {model_name }: {e }")
            return None ,None 

    def get_model_age_hours (self ,model_name :str )->float :
        """Get age of model in hours."""
        meta_path =self .model_dir /f"{model_name }_meta.json"
        if not meta_path .exists ():
            return float ('inf')

        try :
            with open (meta_path ,'r')as f :
                metadata =json .load (f )
            saved_at =metadata .get ('saved_at',0 )
            return (time .time ()-saved_at )/3600 
        except :
            return float ('inf')





class PerformanceModel:
    """Predicts encoding speed (FPS) based on video features using RandomForest."""

    def __init__(self, model_persistence: ModelPersistence = None):
        self.model = None
        self.is_trained = False
        self.feature_order = [
            'resolution_pixels', 'source_bitrate_kbps', 'bitrate_per_pixel',
            'complexity_score', 'scenes_per_minute', 'is_nvenc', 'preset_num',
            'frame_rate', 'is_10bit', 'is_hdr'
        ]
        self.model_persistence = model_persistence or ModelPersistence()
        self.training_metadata = {}
        self._load_or_init_model()

    def _load_or_init_model(self):
        """Try to load existing model or initialize new one."""
        model, metadata = self.model_persistence.load_model('performance_model')
        if model is not None:
            self.model = model
            self.training_metadata = metadata
            self.is_trained = True
            print(f"Loaded existing PerformanceModel trained on {metadata.get('num_samples', 0)} samples")
        else:
            # Changed from LinearRegression to RandomForestRegressor
            self.model = RandomForestRegressor(
                n_estimators=50,
                random_state=42,
                max_depth=10,
                n_jobs=-1
            )
            self.is_trained = False

    def train(self, db_records: list):
        """Train the model on historical performance data."""
        if len(db_records) < 15:
            print(f"PerformanceModel: Not enough data ({len(db_records)} records). Need at least 15.")
            return

        features = []
        targets = []

        for record in db_records:
            try:
                if not record.get('final_encode_fps') or record['final_encode_fps'] <= 0:
                    continue
                if record.get('skipped_on_failure'):
                    continue

                feature_set = {}
                for feature in self.feature_order:
                    if feature == 'is_nvenc':
                        feature_set[feature] = 1 if record.get('encoder_type') == 'nvenc' else 0
                    elif feature == 'preset_num':
                        preset = str(record.get('preset', '5'))
                        if preset.startswith('p'):
                            feature_set[feature] = int(preset[1:]) if preset[1:].isdigit() else 5
                        else:
                            feature_set[feature] = int(preset) if preset.isdigit() else 5
                    else:
                        feature_set[feature] = float(record.get(feature, 0) or 0)

                feature_vector = [feature_set[f] for f in self.feature_order]
                features.append(feature_vector)
                targets.append(record['final_encode_fps'])

            except Exception as e:
                continue

        if len(features) < 15:
            print(f"PerformanceModel: Only {len(features)} valid records after filtering.")
            return

        X = np.array(features)
        y = np.array(targets)

        # Changed from LinearRegression to RandomForestRegressor
        self.model = RandomForestRegressor(
            n_estimators=50,
            random_state=42,
            max_depth=10,
            n_jobs=-1
        )
        self.model.fit(X, y)
        self.is_trained = True

        train_score = self.model.score(X, y)
        self.training_metadata = {
            'num_samples': len(X),
            'train_score': train_score,
            'mean_fps': float(np.mean(y)),
            'std_fps': float(np.std(y)),
            'trained_at': time.time()
        }

        self.model_persistence.save_model(self.model, 'performance_model', self.training_metadata)
        print(f"PerformanceModel: Trained on {len(X)} samples, R² = {train_score:.3f}")

    def update_model_incrementally(self, new_records: list, max_samples: int = 500):
        """Update model with recent data without full retrain."""
        if len(new_records) < 5:
            return False
        
        # Get existing training data
        if database_manager:
            existing_records = database_manager.get_all_performance_records(limit=max_samples)
            # Combine with new records (new ones first)
            all_records = new_records + existing_records[:max_samples - len(new_records)]
            
            # Retrain
            self.train(all_records)
            print(f"Performance model updated with {len(new_records)} new samples")
            return True
        return False

    def predict_fps(self, file_features: dict, encoder_features: dict = None) -> tuple[float, str]:
        """Predict encoding FPS with confidence level."""
        all_features = file_features.copy()
        if encoder_features:
            all_features.update(encoder_features)

        if not self.is_trained:
            base_fps = 200 if all_features.get('is_nvenc') else 50
            resolution_factor = (1920 * 1080) / max(all_features.get('resolution_pixels', 1920 * 1080), 1)
            complexity_factor = 1.0 - (all_features.get('complexity_score', 0.5) * 0.5)
            estimated_fps = base_fps * resolution_factor * complexity_factor
            return max(estimated_fps, 5.0), 'low'

        try:
            feature_vector = np.array([[all_features.get(f, 0) for f in self.feature_order]])
            
            # Get predictions from all trees for confidence assessment
            if hasattr(self.model, 'estimators_'):
                tree_predictions = [est.predict(feature_vector)[0] for est in self.model.estimators_]
                predicted_fps = np.mean(tree_predictions)
                prediction_std = np.std(tree_predictions)
                
                # Determine confidence based on prediction variance
                if prediction_std < predicted_fps * 0.1:  # Less than 10% variance
                    confidence = 'high'
                elif prediction_std < predicted_fps * 0.25:  # Less than 25% variance
                    confidence = 'medium'
                else:
                    confidence = 'low'
            else:
                predicted_fps = self.model.predict(feature_vector)[0]
                confidence = 'medium'

            # Clip to reasonable ranges
            if all_features.get('is_nvenc'):
                predicted_fps = np.clip(predicted_fps, 10, 1000)
            else:
                predicted_fps = np.clip(predicted_fps, 1, 200)

            return max(predicted_fps, 5.0), confidence

        except Exception as e:
            print(f"PerformanceModel prediction error: {e}")
            base_fps = 200 if all_features.get('is_nvenc') else 50
            return base_fps, 'low'



class SamplingTimePredictor:
    """Predicts sample creation time using a simple linear model."""

    def __init__(self, model_persistence: ModelPersistence = None):
        self.model = None
        self.is_trained = False
        self.feature_order = [
            'total_sample_duration_s', 'resolution_pixels', 
            'source_bitrate_kbps', 'is_nvenc_sample_encoder'
        ]
        self.model_persistence = model_persistence or ModelPersistence()
        self.training_metadata = {}
        self._load_or_init_model()

    def _load_or_init_model(self):
        model, metadata = self.model_persistence.load_model('sampling_time_model')
        if model is not None:
            self.model = model
            self.training_metadata = metadata
            self.is_trained = True
            print(f"Loaded existing SamplingTimePredictor trained on {metadata.get('num_samples', 0)} samples")
        else:
            self.model = LinearRegression()
            self.is_trained = False

    def train(self, db_records: list):
        if len(db_records) < 20:
            print(f"SamplingTimePredictor: Not enough data ({len(db_records)} records). Need at least 20.")
            return

        features = []
        targets = []

        for record in db_records:
            try:
                if not record.get('sample_creation_time') or record['sample_creation_time'] <= 0:
                    continue

                feature_set = {
                    'total_sample_duration_s': float(record.get('total_sample_duration_s', 0)),
                    'resolution_pixels': float(record.get('resolution_pixels', 0)),
                    'source_bitrate_kbps': float(record.get('source_bitrate_kbps', 0)),
                    'is_nvenc_sample_encoder': 1 if record.get('master_sample_encoder') == 'nvenc' else 0
                }
                
                feature_vector = [feature_set[f] for f in self.feature_order]
                features.append(feature_vector)
                targets.append(record['sample_creation_time'])
            except (ValueError, TypeError):
                continue

        if len(features) < 20:
            return

        X = np.array(features)
        y = np.array(targets)
        
        self.model.fit(X, y)
        self.is_trained = True
        
        train_score = self.model.score(X, y)
        self.training_metadata = {'num_samples': len(X), 'train_score': train_score}
        self.model_persistence.save_model(self.model, 'sampling_time_model', self.training_metadata)
        print(f"SamplingTimePredictor: Trained on {len(X)} samples, R² = {train_score:.3f}")

    def predict(self, features_dict: dict) -> float:
        if not self.is_trained:
            # Fallback heuristic if model is not trained
            return 15.0 + (features_dict.get('total_sample_duration_s', 12) * 1.5)

        try:
            feature_vector = np.array([[features_dict.get(f, 0) for f in self.feature_order]])
            prediction = self.model.predict(feature_vector)[0]
            return max(prediction, 5.0)  # Ensure a minimum predicted time
        except Exception:
            return 30.0 # Safe fallback on error


class SearchTimePredictor:
    """Predicts CQ search time using a simple linear model."""

    def __init__(self, model_persistence: ModelPersistence = None):
        self.model = None
        self.is_trained = False
        self.feature_order = [
            'search_iterations', 'resolution_pixels', 'total_sample_duration_s'
        ]
        self.model_persistence = model_persistence or ModelPersistence()
        self.training_metadata = {}
        self._load_or_init_model()

    def _load_or_init_model(self):
        model, metadata = self.model_persistence.load_model('search_time_model')
        if model is not None:
            self.model = model
            self.training_metadata = metadata
            self.is_trained = True
            print(f"Loaded existing SearchTimePredictor trained on {metadata.get('num_samples', 0)} samples")
        else:
            self.model = LinearRegression()
            self.is_trained = False

    def train(self, db_records: list):
        if len(db_records) < 20:
            print(f"SearchTimePredictor: Not enough data ({len(db_records)} records). Need at least 20.")
            return

        features = []
        targets = []

        for record in db_records:
            try:
                if not record.get('quality_search_time') or record['quality_search_time'] <= 0 or not record.get('search_iterations'):
                    continue

                feature_set = {
                    'search_iterations': int(record.get('search_iterations', 0)),
                    'resolution_pixels': float(record.get('resolution_pixels', 0)),
                    'total_sample_duration_s': float(record.get('total_sample_duration_s', 0))
                }
                
                feature_vector = [feature_set[f] for f in self.feature_order]
                features.append(feature_vector)
                targets.append(record['quality_search_time'])
            except (ValueError, TypeError):
                continue
        
        if len(features) < 20:
            return

        X = np.array(features)
        y = np.array(targets)
        
        self.model.fit(X, y)
        self.is_trained = True
        
        train_score = self.model.score(X, y)
        self.training_metadata = {'num_samples': len(X), 'train_score': train_score}
        self.model_persistence.save_model(self.model, 'search_time_model', self.training_metadata)
        print(f"SearchTimePredictor: Trained on {len(X)} samples, R² = {train_score:.3f}")

    def predict(self, features_dict: dict) -> float:
        if not self.is_trained:
            # Fallback heuristic if model is not trained
            return 20.0 + (features_dict.get('search_iterations', 2) * 15.0)

        try:
            feature_vector = np.array([[features_dict.get(f, 0) for f in self.feature_order]])
            prediction = self.model.predict(feature_vector)[0]
            return max(prediction, 10.0) # Ensure a minimum predicted time
        except Exception:
            return 45.0 # Safe fallback on error











class QualityModel:

    
    def __init__(self, encoder_type: str, metric_name: str = 'vmaf', metric_subtype: str = None, model_persistence: ModelPersistence = None):
        self.encoder_type = encoder_type.lower()
        self.metric_name = metric_name.lower()
        self.metric_subtype = metric_subtype
        self.model = None
        self.is_trained = False
        self.feature_order = [
            'cq', 'resolution_pixels', 'source_bitrate_kbps',
            'bitrate_per_pixel', 'complexity_score', 'scenes_per_minute',
            'frame_rate', 'is_10bit', 'is_hdr',
            # Added encoder-specific features
            'is_nvenc', 'preset_num'
        ]
        self.model_persistence = model_persistence or ModelPersistence()
        self.training_metadata = {}
        self._load_or_init_model()

    def _get_model_name(self) -> str:
        """Get the model name for this specific encoder, metric, and subtype."""
        base_name = f'quality_model_{self.encoder_type}_{self.metric_name}'
        if self.metric_name == 'vmaf' and self.metric_subtype:
            return f'{base_name}_{self.metric_subtype}'
        return base_name

    def _load_or_init_model(self):
        """Try to load existing model or initialize new one."""
        model_name = self._get_model_name()
        model, metadata = self.model_persistence.load_model(model_name)
        if model is not None:
            self.model = model
            self.training_metadata = metadata
            self.is_trained = True
            subtype_info = f" ({self.metric_subtype})" if self.metric_subtype else ""
            print(f"Loaded existing QualityModel for {self.encoder_type.upper()}/{self.metric_name.upper()}{subtype_info} trained on {metadata.get('num_samples', 0)} samples")
        else:
            self.model = RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1, max_depth=10)
            self.is_trained = False
            subtype_info = f" ({self.metric_subtype})" if self.metric_subtype else ""
            print(f"Initialized new QualityModel for {self.encoder_type.upper()}/{self.metric_name.upper()}{subtype_info}")

    def train(self, all_quality_records: list):
        """Train the model on historical quality test data for this specific encoder, metric, and subtype."""
        # 1. Filter records for the current encoder
        encoder_records = [r for r in all_quality_records if r.get('encoder_type', '').lower() == self.encoder_type]

        # 2. Filter by metric and subtype
        if self.metric_name == 'vmaf' and self.metric_subtype:
            metric_records = [r for r in encoder_records
                              if r.get('metric_name', '').lower() == self.metric_name
                              and r.get('metric_subtype', '').lower() == self.metric_subtype.lower()]
        else:
            metric_records = [r for r in encoder_records
                              if r.get('metric_name', '').lower() == self.metric_name]

        model_id = f"{self.encoder_type.upper()}/{self.metric_name.upper()}" + (f" ({self.metric_subtype})" if self.metric_subtype else "")
        if len(metric_records) < 50:
            print(f"QualityModel ({model_id}): Not enough data ({len(metric_records)} records). Need at least 50.")
            return

        features = []
        targets = []

        for record in metric_records:
            try:
                if not record.get('score') or record['score'] <= 0 or not record.get('cq'):
                    continue

                feature_set = {k: float(v or 0) for k, v in record.items() if k in self.feature_order and k not in ['is_nvenc', 'preset_num']}
                
                # Manually add encoder features
                feature_set['is_nvenc'] = 1 if record.get('encoder_type') == 'nvenc' else 0
                preset = str(record.get('preset','5'))
                if preset.startswith('p'):
                    feature_set['preset_num'] = int(preset[1:]) if preset[1:].isdigit() else 5
                else:
                    feature_set['preset_num'] = int(preset) if preset.isdigit() else 5

                feature_vector = [feature_set.get(f, 0) for f in self.feature_order]
                features.append(feature_vector)
                targets.append(record['score'])
            except (ValueError, TypeError):
                continue

        if len(features) < 50:
            print(f"QualityModel ({model_id}): Only {len(features)} valid records after filtering.")
            return

        X = np.array(features)
        y = np.array(targets)
        split_idx = int(len(X) * 0.8)
        X_train, X_val = X[:split_idx], X[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]

        self.model = RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1, max_depth=10)
        self.model.fit(X_train, y_train)
        self.is_trained = True

        train_score = self.model.score(X_train, y_train)
        val_score = self.model.score(X_val, y_val) if len(X_val) > 0 else train_score

        self.training_metadata = {
            'num_samples': len(X), 'train_score': train_score, 'val_score': val_score,
            'mean_score': float(np.mean(y)), 'std_score': float(np.std(y)),
            'encoder_type': self.encoder_type, 'metric_name': self.metric_name, 'metric_subtype': self.metric_subtype,
            'trained_at': time.time()
        }

        model_name = self._get_model_name()
        self.model_persistence.save_model(self.model, model_name, self.training_metadata)
        print(f"QualityModel ({model_id}): Trained on {len(X)} samples, Train R² = {train_score:.3f}, Val R² = {val_score:.3f}")



    def predict_with_confidence(self, file_features: dict, encoder_features: dict, 
                               target_score: float, tolerance: float, 
                               cq_min: int, cq_max: int) -> tuple:
        """Predict optimal CQ with confidence assessment."""
        if not self.is_trained:
            return None, None, 'none', {}
        
        try:
            # Get predictions from all trees for confidence assessment
            predictions = {}
            all_trees_predictions = []
            
            for cq in range(cq_min, cq_max + 1):
                feature_dict = {**file_features, **encoder_features, 'cq': cq}
                feature_vector = np.array([[feature_dict.get(f, 0) for f in self.feature_order]])
                
                # Get predictions from each tree
                tree_predictions = [est.predict(feature_vector)[0] for est in self.model.estimators_]
                predictions[cq] = {
                    'mean': np.mean(tree_predictions),
                    'std': np.std(tree_predictions)
                }
                all_trees_predictions.append(tree_predictions)
            
            # Find CQ values within tolerance
            valid_cqs = []
            for cq, pred in predictions.items():
                if target_score - tolerance <= pred['mean'] <= target_score + tolerance:
                    valid_cqs.append(cq)
            
            if not valid_cqs:
                # Find closest if none in range
                best_cq = min(predictions.keys(), 
                             key=lambda x: abs(predictions[x]['mean'] - target_score))
            else:
                # Get highest CQ (best compression) within tolerance
                best_cq = max(valid_cqs)
            
            # Calculate confidence based on prediction variance
            prediction_std = predictions[best_cq]['std']
            
            # Also check consistency across neighboring CQs
            if best_cq > cq_min and best_cq < cq_max:
                neighbor_scores = [
                    predictions.get(best_cq - 1, {}).get('mean', 0),
                    predictions[best_cq]['mean'],
                    predictions.get(best_cq + 1, {}).get('mean', 0)
                ]
                score_gradient = np.diff(neighbor_scores)
                gradient_consistency = np.std(score_gradient)
            else:
                gradient_consistency = 1.0
            
            # Determine confidence level
            if prediction_std < 1.0 and gradient_consistency < 0.5:
                confidence = 'high'
            elif prediction_std < 2.0 and gradient_consistency < 1.0:
                confidence = 'medium'
            else:
                confidence = 'low'
            
            return best_cq, predictions[best_cq]['mean'], confidence, predictions
            
        except Exception as e:
            print(f"Error in predict_with_confidence: {e}")
            return None, None, 'none', {}






    def predict_score(self, file_features: dict, encoder_features: dict, cq: int) -> tuple[float, float]:
        """Predict quality score with confidence interval."""
        if not self.is_trained:
            return None, None
        
        try:
            feature_dict = {**file_features, **encoder_features, 'cq': cq}
            feature_vector = np.array([[feature_dict.get(f, 0) for f in self.feature_order]])
            
            predictions = np.array([est.predict(feature_vector)[0] for est in self.model.estimators_])
            return np.mean(predictions), np.std(predictions)
        except Exception as e:
            model_id = f"{self.encoder_type.upper()}/{self.metric_name.upper()}"
            print(f"QualityModel ({model_id}) prediction error: {e}")
            return None, None

    def predict_cq_curve(self, file_features: dict, encoder_features: dict, cq_min: int, cq_max: int) -> dict:
        """Predict quality scores for all CQ values in range."""
        if not self.is_trained: return {}
        return {cq: {'score': s, 'confidence': c} for cq in range(cq_min, cq_max + 1) if (s_c := self.predict_score(file_features, encoder_features, cq)) and (s := s_c[0]) is not None and (c := s_c[1]) is not None}

    def suggest_cq_range(self, file_features: dict, encoder_features: dict, target_score: float,
                         tolerance: float, cq_min: int, cq_max: int) -> tuple[int, int, dict]:
        """Suggest narrowed CQ range likely to contain target score."""
        if not self.is_trained:
            return cq_min, cq_max, {}
        
        predictions = self.predict_cq_curve(file_features, encoder_features, cq_min, cq_max)
        if not predictions:
            return cq_min, cq_max, {}
        
        candidates = [cq for cq, pred in predictions.items() if target_score - tolerance <= pred['score'] <= target_score + tolerance]
        
        if candidates:
            suggested_min = max(cq_min, min(candidates) - 2)
            suggested_max = min(cq_max, max(candidates) + 2)
        else:
            best_cq = min(predictions.keys(), key=lambda x: abs(predictions[x]['score'] - target_score))
            suggested_min = max(cq_min, best_cq - 5)
            suggested_max = min(cq_max, best_cq + 5)
            
        return suggested_min, suggested_max, predictions




class DatabaseManager :
    def __init__ (self ,db_path :str ):
        self .db_path =db_path 
        self ._local =threading .local ()
        self ._init_db_once ()
        self ._file_hashes ={}
        self ._file_hashes_lock =threading .Lock ()
        self .feature_extractor =FeatureExtractor ()

    def _get_conn (self ):
        """Creates and returns a new connection for the current thread if one doesn't exist."""
        if not hasattr (self ._local ,"conn")or self ._local .conn is None :
            try :
                conn =sqlite3 .connect (self .db_path ,check_same_thread =False ,timeout =10.0 )
                conn .row_factory =sqlite3 .Row 
                self ._local .conn =conn 
            except Exception as e :
                print (f"CRITICAL: Failed to create thread-local database connection: {e }")
                self ._local .conn =None 
        return self ._local .conn 

    def _init_db_once (self ):
        """Creates tables with enhanced schema for ML features."""
        conn =None 
        try :
            conn =sqlite3 .connect (self .db_path ,timeout =10.0 )


            conn .execute ('''
                CREATE TABLE IF NOT EXISTS quality_cache (
                    sample_key TEXT,
                    cq INTEGER,
                    metric_name TEXT,
                    metric_subtype TEXT,
                    score REAL,
                    timestamp REAL,
                    file_hash TEXT,
                    features_json TEXT,
                    PRIMARY KEY (sample_key, cq, metric_name, metric_subtype)
                )
            ''')

            conn .execute ('''
                CREATE TABLE IF NOT EXISTS performance_log (
                    file_hash TEXT PRIMARY KEY,
                    batch_id TEXT,
                    worker_start_timestamp REAL,
                    source_filename TEXT,
                    source_codec TEXT,
                    source_pixel_format TEXT,
                    resolution_key TEXT,
                    bitrate_per_pixel REAL,
                    encoder_type TEXT,
                    preset TEXT,
                    output_bit_depth_setting TEXT,
                    quality_metric TEXT,
                    target_score REAL,
                    final_score REAL,
                    final_bitrate_kbps REAL,
                    best_cq INTEGER,
                    cq_search_path TEXT,
                    search_iterations INTEGER,
                    sample_creation_time REAL,
                    num_samples_requested INTEGER,
                    sample_duration_s INTEGER,
                    total_sample_duration_s REAL,
                    quality_search_time REAL,
                    final_encode_fps REAL,
                    size_before_mb REAL,
                    size_after_mb REAL,
                    skipped_on_failure INTEGER,
                    cpu_name TEXT,
                    gpu_name TEXT,
                    ffmpeg_version TEXT,
                    frame_rate REAL,
                    video_duration_seconds REAL,
                    aspect_ratio REAL,
                    container_format TEXT,
                    source_bitrate_kbps REAL,
                    complexity_score REAL,
                    sampling_method TEXT,
                    prediction_error_ratio REAL,
                    resolution_pixels INTEGER,
                    width INTEGER,
                    height INTEGER,
                    scenes_per_minute REAL,
                    scene_count INTEGER,
                    avg_scene_duration REAL,
                    preset_num INTEGER,
                    is_10bit INTEGER,
                    is_hdr INTEGER,
                    features_json TEXT,
                    timestamp REAL
                )
            ''')

            conn .execute ('''
                CREATE TABLE IF NOT EXISTS ml_metadata (
                    model_name TEXT PRIMARY KEY,
                    last_trained REAL,
                    num_samples INTEGER,
                    train_score REAL,
                    val_score REAL,
                    metadata_json TEXT
                )
            ''')


            cursor =conn .execute ("PRAGMA table_info(quality_cache)")
            columns =[c [1 ]for c in cursor .fetchall ()]

            if 'features_json'not in columns :
                conn .execute ("ALTER TABLE quality_cache ADD COLUMN features_json TEXT")
                print ("Migrated database: Added 'features_json' column to 'quality_cache'.")

            cursor =conn .execute ("PRAGMA table_info(performance_log)")
            columns =[c [1 ]for c in cursor .fetchall ()]

            if 'features_json'not in columns :
                conn .execute ("ALTER TABLE performance_log ADD COLUMN features_json TEXT")
                print ("Migrated database: Added 'features_json' column to 'performance_log'.")

            conn .execute ('CREATE INDEX IF NOT EXISTS idx_sample_key ON quality_cache(sample_key)')
            conn .execute ('CREATE INDEX IF NOT EXISTS idx_qc_file_hash ON quality_cache(file_hash)')
            conn .execute ('CREATE INDEX IF NOT EXISTS idx_perf_ml ON performance_log(encoder_type, resolution_key, preset, skipped_on_failure)')
            conn .execute ('CREATE INDEX IF NOT EXISTS idx_timestamp ON performance_log(timestamp DESC)')

            conn .commit ()
            print ("Database initialized successfully with ML-enhanced schema")

        except Exception as e :
            print (f"CRITICAL: Database initialization failed: {e }")
        finally :
            if conn :
                conn .close ()

    def _get_file_hash (self ,file_path :str )->str :
        """Calculate and cache file hash."""
        with self ._file_hashes_lock :
            if file_path in self ._file_hashes :
                return self ._file_hashes [file_path ]

            h =hashlib .sha256 ()
            with open (file_path ,'rb')as f :
                while chunk :=f .read (8192 *4 ):
                    h .update (chunk )
            file_hash =h .hexdigest ()
            self ._file_hashes [file_path ]=file_hash 
            return file_hash 



    def store_prediction_error(self, file_hash: str, predicted_cq: int, optimal_cq: int,
                              predicted_score: float, actual_score: float, features_json: str):
        """Store ML prediction errors for learning."""
        conn = self._get_conn()
        if not conn:
            return
        
        try:
            # Create table if it doesn't exist
            conn.execute('''
                CREATE TABLE IF NOT EXISTS ml_prediction_errors (
                    file_hash TEXT,
                    predicted_cq INTEGER,
                    optimal_cq INTEGER,
                    predicted_score REAL,
                    actual_score REAL,
                    cq_error INTEGER,
                    score_error REAL,
                    features_json TEXT,
                    timestamp REAL,
                    PRIMARY KEY (file_hash, timestamp)
                )
            ''')
            
            conn.execute('''
                INSERT INTO ml_prediction_errors VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                file_hash, predicted_cq, optimal_cq, predicted_score, actual_score,
                optimal_cq - predicted_cq, actual_score - predicted_score,
                features_json, time.time()
            ))
            
            conn.commit()
        except Exception as e:
            print(f"Error storing prediction error: {e}")
    
    def get_recent_errors_for_signature(self, content_signature: str, limit: int = 20) -> List[Dict]:
        """Get recent prediction errors for a content signature."""
        conn = self._get_conn()
        if not conn:
            return []
        
        try:
            # This would need the content signature stored, for now return recent errors
            cursor = conn.execute('''
                SELECT * FROM ml_prediction_errors 
                ORDER BY timestamp DESC 
                LIMIT ?
            ''', (limit,))
            
            return [dict(row) for row in cursor]
        except Exception as e:
            print(f"Error retrieving prediction errors: {e}")
            return []






    def _get_sample_key(self, source_video_path: str, sample_timestamps: List[float],
                          complexity_score: float = None, encoder_settings: dict = None) -> str:
        """
        Generate a unique key for a sample that is specific to the encoder settings.
        """
        source_hash = self._get_file_hash(source_video_path)
        timestamps_str = str(sorted(sample_timestamps))
        complexity_str = f"-c{complexity_score:.1f}" if complexity_score is not None else ""
        
        # --- NEW: Create a string from the relevant encoder settings ---
        settings_str = ""
        if encoder_settings:
            preset = encoder_settings.get('preset', '')
            quality_mode = encoder_settings.get('quality_mode', '')
            settings_str = f"-{preset}-{quality_mode}" if quality_mode else f"-{preset}"

        combined_string = f"{source_hash}-{timestamps_str}{complexity_str}{settings_str}"
        return hashlib.sha256(combined_string.encode()).hexdigest()

    def get_quality_cache(self, source_video_path: str, sample_timestamps: List[float],
                         cq: int, metric_name: str, complexity_score: float = None,
                         metric_subtype: str = None, encoder_settings: dict = None) -> Optional[float]:
        """Retrieve cached quality score using a settings-aware key."""
        conn = self._get_conn()
        if not conn:
            return None
        
        # Pass the encoder settings to the key generation function
        sample_key = self._get_sample_key(source_video_path, sample_timestamps, complexity_score, encoder_settings)
        
        if metric_name == 'vmaf' and not metric_subtype:
            return None
        
        if metric_name != 'vmaf':
            metric_subtype = metric_name
        
        try:
            cursor = conn.execute(
                'SELECT score FROM quality_cache WHERE sample_key=? AND cq=? AND metric_name=? AND metric_subtype=?',
                (sample_key, cq, metric_name, metric_subtype)
            )
            result = cursor.fetchone()
            return result['score'] if result else None
        except Exception as e:
            print(f"Error retrieving quality cache: {e}")
            return None 

    def set_quality_cache(self, source_video_path: str, sample_timestamps: List[float],
                         cq: int, metric_name: str, score: float,
                         complexity_score: float = None, features: dict = None,
                         metric_subtype: str = None, encoder_settings: dict = None):
        """Store quality test result with a settings-aware key."""
        conn = self._get_conn()
        if not conn:
            return
        
        # Pass the encoder settings to the key generation function
        sample_key = self._get_sample_key(source_video_path, sample_timestamps, complexity_score, encoder_settings)
        file_hash = self._get_file_hash(source_video_path)
        features_json = json.dumps(features) if features else None
        
        if metric_name == 'vmaf' and not metric_subtype:
            print(f"Warning: VMAF cache entry missing subtype, skipping cache")
            return
        
        if metric_name != 'vmaf':
            metric_subtype = metric_name
        
        try:
            conn.execute(
                'INSERT OR REPLACE INTO quality_cache VALUES (?, ?, ?, ?, ?, ?, ?, ?)',
                (sample_key, cq, metric_name, metric_subtype, score, time.time(), file_hash, features_json)
            )
            conn.commit()
        except Exception as e:
            print(f"Error setting quality cache: {e}")


    def log_performance (self ,metrics :Dict ):
        """Log performance metrics with full feature set."""
        conn =self ._get_conn ()
        if not conn :
            return 

        columns =[
        'file_hash','batch_id','worker_start_timestamp','source_filename',
        'source_codec','source_pixel_format','resolution_key','bitrate_per_pixel',
        'encoder_type','preset','output_bit_depth_setting','quality_metric',
        'target_score','final_score','final_bitrate_kbps','best_cq',
        'cq_search_path','search_iterations','sample_creation_time',
        'num_samples_requested','sample_duration_s','total_sample_duration_s',
        'quality_search_time','final_encode_fps','size_before_mb','size_after_mb',
        'skipped_on_failure','cpu_name','gpu_name','ffmpeg_version','frame_rate',
        'video_duration_seconds','aspect_ratio','container_format','source_bitrate_kbps',
        'complexity_score','sampling_method','prediction_error_ratio',
        'resolution_pixels','width','height','scenes_per_minute','scene_count',
        'avg_scene_duration','preset_num','is_10bit','is_hdr','features_json','timestamp'
        ]


        if 'features_json'not in metrics :
            features ={}
            for key in ['resolution_pixels','source_bitrate_kbps','bitrate_per_pixel',
            'complexity_score','scenes_per_minute','frame_rate','is_10bit','is_hdr']:
                if key in metrics :
                    features [key ]=metrics [key ]
            metrics ['features_json']=json .dumps (features )if features else None 

        values =[metrics .get (col )for col in columns [:-1 ]]
        values .append (time .time ())

        placeholders =', '.join (['?']*len (columns ))
        sql =f'INSERT OR REPLACE INTO performance_log ({", ".join (columns )}) VALUES ({placeholders })'

        try :
            conn .execute (sql ,tuple (values ))
            conn .commit ()
        except Exception as e :
            print (f"Error logging performance: {e }")

    def get_all_performance_records (self ,limit :int =1000 )->List [Dict ]:
        """Retrieve recent performance records for ML training."""
        conn =self ._get_conn ()
        if not conn :
            return []

        try :
            cursor =conn .execute ('''
                SELECT * FROM performance_log 
                WHERE final_encode_fps IS NOT NULL 
                AND final_encode_fps > 0
                AND (skipped_on_failure IS NULL OR skipped_on_failure = 0)
                ORDER BY timestamp DESC 
                LIMIT ?
            ''',(limit ,))

            records =[]
            for row in cursor :
                record =dict (row )

                if record .get ('features_json'):
                    try :
                        features =json .loads (record ['features_json'])
                        record .update (features )
                    except :
                        pass 
                records .append (record )

            return records 

        except Exception as e :
            print (f"Error retrieving performance records: {e }")
            return []

    def get_quality_records_with_features (self ,limit :int =5000 )->List [Dict ]:
        """Retrieve quality cache entries with associated file features."""
        conn =self ._get_conn ()
        if not conn :
            return []

        try :

            cursor =conn .execute ('''
                SELECT 
                    q.cq, q.score, q.metric_name,
                    p.resolution_pixels, p.source_bitrate_kbps, p.bitrate_per_pixel,
                    p.complexity_score, p.scenes_per_minute, p.frame_rate,
                    q.features_json as q_features, p.features_json as p_features
                FROM quality_cache q
                LEFT JOIN performance_log p ON q.file_hash = p.file_hash
                WHERE q.score IS NOT NULL AND q.score > 0
                ORDER BY q.timestamp DESC
                LIMIT ?
            ''',(limit ,))

            records =[]
            for row in cursor :
                record =dict (row )


                features ={}
                if record .get ('p_features'):
                    try :
                        features .update (json .loads (record ['p_features']))
                    except :
                        pass 

                if record .get ('q_features'):
                    try :
                        features .update (json .loads (record ['q_features']))
                    except :
                        pass 

                record .update (features )


                record .pop ('q_features',None )
                record .pop ('p_features',None )

                records .append (record )

            return records 

        except Exception as e :
            print (f"Error retrieving quality records: {e }")
            return []



    def get_quality_records_for_metric(self, metric_name: str, metric_subtype: str = None, limit: int = 5000) -> List[Dict]:
        """Retrieve quality cache entries for a specific metric and subtype with associated file and ENCODER features."""
        conn = self._get_conn()
        if not conn:
            return []
        
        # Base query now selects encoder_type and preset from the performance log
        sql_query = '''
            SELECT 
                q.cq, q.score, q.metric_name, q.metric_subtype,
                p.resolution_pixels, p.source_bitrate_kbps, p.bitrate_per_pixel,
                p.complexity_score, p.scenes_per_minute, p.frame_rate,
                p.is_10bit, p.is_hdr, p.encoder_type, p.preset,
                q.features_json as q_features, p.features_json as p_features
            FROM quality_cache q
            LEFT JOIN performance_log p ON q.file_hash = p.file_hash
            WHERE q.score IS NOT NULL AND q.score > 0
        '''
        
        params = []
        
        try:
            # Add filtering for metric and subtype
            if metric_subtype:
                sql_query += ' AND LOWER(q.metric_name) = LOWER(?) AND LOWER(q.metric_subtype) = LOWER(?)'
                params.extend([metric_name, metric_subtype])
            else:
                sql_query += ' AND LOWER(q.metric_name) = LOWER(?)'
                params.append(metric_name)
            
            sql_query += ' ORDER BY q.timestamp DESC LIMIT ?'
            params.append(limit)

            cursor = conn.execute(sql_query, tuple(params))
            
            records = []
            for row in cursor:
                record = dict(row)
                
                features = {}
                if record.get('p_features'):
                    try: features.update(json.loads(record['p_features']))
                    except: pass
                
                if record.get('q_features'):
                    try: features.update(json.loads(record['q_features']))
                    except: pass
                
                record.update(features)
                record.pop('q_features', None)
                record.pop('p_features', None)
                
                records.append(record)
            
            return records
        
        except Exception as e:
            print(f"Error retrieving quality records for metric {metric_name} (subtype: {metric_subtype}): {e}")
            return []





    def get_all_quality_records_for_training(self, limit: int = 20000) -> List[Dict]:
        """
        Retrieves all self-contained records from quality_cache for ML training.
        This method is designed to be self-sufficient, parsing the features_json
        to provide a complete record without needing to join performance_log.
        """
        conn = self._get_conn()
        if not conn: return []
        
        try:
            cursor = conn.execute(
                'SELECT cq, score, metric_name, metric_subtype, features_json FROM quality_cache WHERE score IS NOT NULL AND features_json IS NOT NULL ORDER BY timestamp DESC LIMIT ?',
                (limit,)
            )
            
            records = []
            for row in cursor:
                record = dict(row)
                try:
                    features = json.loads(record['features_json'])
                    record.update(features)
                    
                    # --- COMPATIBILITY LOGIC ---
                    # Reconstruct fields expected by QualityModel.train to avoid breaking it
                    if 'is_nvenc' in features:
                        is_nvenc = features['is_nvenc']
                        record['encoder_type'] = 'nvenc' if is_nvenc else 'svt_av1'
                        
                        preset_num = features.get('preset_num')
                        if preset_num is not None:
                            if is_nvenc:
                                record['preset'] = f'p{int(preset_num)}'
                            else:
                                record['preset'] = str(int(preset_num))

                    record.pop('features_json', None)
                    records.append(record)
                except (json.JSONDecodeError, TypeError, KeyError):
                    continue # Skip malformed records
                    
            return records
            
        except Exception as e:
            print(f"Error retrieving all quality records for training: {e}")
            return []






    def calculate_file_similarity (self ,file_metrics :Dict ,db_record :Dict )->float :
        """Calculate similarity between two video files."""
        similarity =1.0 


        if file_metrics .get ('resolution_key')!=db_record .get ('resolution_key'):
            similarity *=0.3 


        file_fps =file_metrics .get ('frame_rate',30 )
        db_fps =db_record .get ('frame_rate',30 )
        if db_fps and db_fps >0 :
            fps_diff =abs (file_fps -db_fps )/max (file_fps ,db_fps ,1 )
            similarity *=max (0.7 ,1.0 -fps_diff )


        file_complexity =file_metrics .get ('complexity_score',0.5 )
        db_complexity =db_record .get ('complexity_score',0.5 )
        if db_complexity is not None :
            complexity_diff =abs (file_complexity -db_complexity )
            similarity *=max (0.8 ,1.0 -complexity_diff *2 )


        file_bpp =file_metrics .get ('bitrate_per_pixel',0.05 )
        db_bpp =db_record .get ('bitrate_per_pixel',0.05 )
        if db_bpp and db_bpp >0 :
            bpp_diff =abs (file_bpp -db_bpp )/max (file_bpp ,db_bpp ,0.001 )
            similarity *=max (0.9 ,1.0 -bpp_diff *0.5 )


        file_aspect =file_metrics .get ('aspect_ratio',1.778 )
        db_aspect =db_record .get ('aspect_ratio',1.778 )
        if db_aspect and db_aspect >0 :
            aspect_diff =abs (file_aspect -db_aspect )/max (file_aspect ,db_aspect ,1 )
            similarity *=max (0.95 ,1.0 -aspect_diff *0.3 )

        return similarity 

    def get_performance_stats (self ,resolution_key :str ,encoder_type :str ,
    preset :str ,file_metrics :Dict )->Dict :
        """Get performance statistics with similarity weighting."""
        conn =self ._get_conn ()
        if not conn :
            return {'encode_fps':0 ,'confidence':'low'}

        try :

            cursor =conn .execute ('''
                SELECT * FROM performance_log
                WHERE resolution_key=? AND encoder_type=? AND preset=?
                AND (skipped_on_failure IS NULL OR skipped_on_failure = 0)
                AND final_encode_fps IS NOT NULL AND final_encode_fps > 0
                ORDER BY timestamp DESC
                LIMIT 50
            ''',(resolution_key ,encoder_type ,preset ))

            records =[dict (row )for row in cursor .fetchall ()]

            if not records :
                return {'encode_fps':0 ,'confidence':'low'}


            weighted_items =[]
            for record in records :
                similarity =self .calculate_file_similarity (file_metrics ,record )
                if similarity >0.5 :
                    weighted_items .append ({
                    'record':record ,
                    'weight':similarity 
                    })

            if not weighted_items :

                avg_fps =sum (r ['final_encode_fps']for r in records )/len (records )
                return {
                'encode_fps':avg_fps ,
                'confidence':'low'
                }


            total_weight =sum (item ['weight']for item in weighted_items )

            weighted_fps =sum (item ['record']['final_encode_fps']*item ['weight']
            for item in weighted_items )/total_weight 

            weighted_error =sum (item ['record'].get ('prediction_error_ratio',0 )*item ['weight']
            for item in weighted_items if item ['record'].get ('prediction_error_ratio')is not None )/total_weight 

            confidence ='high'if len (weighted_items )>=5 else 'medium'

            return {
            'encode_fps':weighted_fps ,
            'avg_prediction_error_ratio':weighted_error ,
            'confidence':confidence ,
            'num_samples':len (weighted_items )
            }

        except Exception as e :
            print (f"Error getting performance stats: {e }")
            return {'encode_fps':0 ,'confidence':'low'}

    def get_performance_record_by_hash (self ,file_hash :str )->Optional [Dict ]:
        """Get complete performance record by file hash."""
        conn =self ._get_conn ()
        if not conn :
            return None 

        try :
            cursor =conn .execute (
            'SELECT * FROM performance_log WHERE file_hash=? ORDER BY timestamp DESC LIMIT 1',
            (file_hash ,)
            )
            result =cursor .fetchone ()
            if result :
                record =dict (result )

                if record .get ('features_json'):
                    try :
                        features =json .loads (record ['features_json'])
                        record .update (features )
                    except :
                        pass 
                return record 
            return None 
        except Exception as e :
            print (f"Error retrieving performance record: {e }")
            return None 

    def update_ml_metadata (self ,model_name :str ,metadata :dict ):
        """Store ML model training metadata."""
        conn =self ._get_conn ()
        if not conn :
            return 

        try :
            conn .execute ('''
                INSERT OR REPLACE INTO ml_metadata VALUES (?, ?, ?, ?, ?, ?)
            ''',(
            model_name ,
            time .time (),
            metadata .get ('num_samples',0 ),
            metadata .get ('train_score',0 ),
            metadata .get ('val_score',0 ),
            json .dumps (metadata )
            ))
            conn .commit ()
        except Exception as e :
            print (f"Error updating ML metadata: {e }")





class MemoryManager :
    def __init__ (self ):
        self .main_process =psutil .Process (os .getpid ())

    def get_all_ffmpeg_processes (self ):
        """Get all FFmpeg processes spawned by this script."""
        try :
            children =self .main_process .children (recursive =True )
            ffmpeg_processes =[p for p in children if 'ffmpeg'in p .name ().lower ()]
            return ffmpeg_processes 
        except :
            return []

    def get_total_memory_usage (self )->int :
        """Get total memory usage of main process + all FFmpeg children."""
        try :
            total =self .main_process .memory_info ().rss 
            for ffmpeg_proc in self .get_all_ffmpeg_processes ():
                try :
                    total +=ffmpeg_proc .memory_info ().rss 
                except (psutil .NoSuchProcess ,psutil .AccessDenied ):
                    continue 
            return total 
        except :
            return self .main_process .memory_info ().rss 

    def get_usage_mb (self )->float :
        return self .get_total_memory_usage ()/(1024 **2 )

    def get_detailed_usage (self )->Dict :
        """Get detailed memory breakdown."""
        try :
            main_memory =self .main_process .memory_info ().rss 
            ffmpeg_processes =self .get_all_ffmpeg_processes ()
            ffmpeg_memory =sum (p .memory_info ().rss for p in ffmpeg_processes if p .is_running ())

            return {
            'main_mb':main_memory /(1024 **2 ),
            'ffmpeg_mb':ffmpeg_memory /(1024 **2 ),
            'total_mb':(main_memory +ffmpeg_memory )/(1024 **2 ),
            'ffmpeg_count':len (ffmpeg_processes )
            }
        except :
            total =self .get_total_memory_usage ()
            return {
            'main_mb':total /(1024 **2 ),
            'ffmpeg_mb':0 ,
            'total_mb':total /(1024 **2 ),
            'ffmpeg_count':0 
            }


memory_manager :MemoryManager 


def get_ffmpeg_env ():
    """Sets up the environment for FFmpeg to find necessary libraries."""
    env =os .environ .copy ()
    ffmpeg_bin_dir =Path (SETTINGS .ffmpeg_path ).parent 
    env ['PATH']=f"{ffmpeg_bin_dir }{os .pathsep }{env .get ('PATH','')}"
    return env 


def get_media_info (video_path :str ,cancel_event :threading .Event =None ,process_registry :dict =None ,worker_id :int =None )->Optional [Dict ]:
    """
    Extracts media information using ffprobe.
    - If cancel_event is provided, it runs in a non-blocking, cancellable mode.
    - Otherwise, it runs in a simple, blocking mode.
    """
    if not os .path .exists (video_path ):
        return None 

    cmd =[
    str (SETTINGS .ffprobe_path ),
    '-v','error',
    '-show_entries',
    'format=duration,bit_rate,format_name,size:stream=codec_name,codec_type,width,height,avg_frame_rate,pix_fmt,color_space,color_primaries,color_transfer',
    '-of','json',video_path 
    ]


    if cancel_event and process_registry is not None and worker_id is not None :
        process =None 
        try :
            # Use startupinfo on Windows to hide console window
            startupinfo = None
            if os.name == 'nt':
                startupinfo = subprocess.STARTUPINFO()
                startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
                startupinfo.wShowWindow = subprocess.SW_HIDE
            
            # MODIFIED: Removed the 'creationflags=SUBPROCESS_FLAGS' argument
            # The startupinfo object is the correct and preferred way to hide windows in a GUI app.
            process =subprocess .Popen (cmd ,stdout =subprocess .PIPE ,stderr =subprocess .PIPE ,
            text =True ,encoding ='utf-8',
            startupinfo=startupinfo ,env =FFMPEG_ENV )
            process_registry [worker_id ]=process 

            while process .poll ()is None :
                if cancel_event .is_set ():
                    print (f"Cancellation detected for worker {worker_id }. Killing ffprobe process {process .pid }.")
                    process .kill ()
                    process .wait ()
                    return None 
                time .sleep (0.1 )

            stdout ,stderr =process .communicate ()
            if process .returncode ==0 and stdout :
                return json .loads (stdout )
            else :
                logging .error (f"FFPROBE failed for {video_path }: {stderr }")
                return None 
        except Exception as e :
            if not cancel_event .is_set ():
                logging .error (f"FFPROBE exception for {video_path }: {e }")
            if process :
                process .kill ()
            return None 
        finally :
            if worker_id in process_registry :
                del process_registry [worker_id ]


    else :
        try :
            # Use startupinfo on Windows to hide console window
            startupinfo = None
            if os.name == 'nt':
                startupinfo = subprocess.STARTUPINFO()
                startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
                startupinfo.wShowWindow = subprocess.SW_HIDE
            
            result =subprocess .run (cmd ,capture_output =True ,text =True ,check =True ,
            encoding ='utf-8',creationflags =SUBPROCESS_FLAGS ,startupinfo=startupinfo ,
            env =FFMPEG_ENV ,timeout =60 )
            if result .stdout :
                return json .loads (result .stdout )
            return None 
        except Exception as e :
            logging .error (f"Simple FFPROBE check failed for {video_path }: {e }")
            return None


def build_color_args (video_stream :Dict )->List [str ]:
    """Builds FFmpeg color-related arguments from a video stream's info."""
    args =[]
    if video_stream .get ('color_space')and video_stream ['color_space']!='unknown':
        args .extend (['-colorspace',video_stream ['color_space']])
    if video_stream .get ('color_primaries')and video_stream ['color_primaries']!='unknown':
        args .extend (['-color_primaries',video_stream ['color_primaries']])
    if video_stream .get ('color_trc')and video_stream ['color_trc']!='unknown':
        args .extend (['-color_trc',video_stream ['color_trc']])
    return args 


def get_target_pix_fmt (source_pix_fmt :str ,encoder_type :str ,output_bit_depth :str )->Optional [str ]:
    """Determines the target pixel format based on user settings and encoder capabilities."""
    if not source_pix_fmt :
        return None 

    is_10bit_source ='10'in source_pix_fmt or '12'in source_pix_fmt 
    is_422_source ='422'in source_pix_fmt 
    is_444_source ='444'in source_pix_fmt 

    if output_bit_depth =='source':
        return 'p010le'if encoder_type =='nvenc'and source_pix_fmt =='yuv420p10le'else source_pix_fmt 

    if output_bit_depth =='8bit':
        if is_444_source :return 'yuv444p'
        if is_422_source :return 'yuv422p'
        return 'yuv420p'

    if output_bit_depth =='10bit':
        if is_10bit_source :
            return 'p010le'if encoder_type =='nvenc'and '420'in source_pix_fmt else source_pix_fmt 
        else :
            if encoder_type =='nvenc':
                return 'p010le'
            else :
                if is_444_source :return 'yuv444p10le'
                if is_422_source :return 'yuv422p10le'
                return 'yuv420p10le'

    return source_pix_fmt 


def build_encoder_args (quality_value :int ,video_stream :Dict ,for_final_encode :bool )->List [str ]:
    """Builds a list of FFmpeg arguments based on the selected encoder and color settings."""
    source_pix_fmt =video_stream .get ('pix_fmt')
    color_args =build_color_args (video_stream )

    if for_final_encode :
        target_pix_fmt =get_target_pix_fmt (source_pix_fmt ,SETTINGS .encoder_type ,SETTINGS .output_bit_depth )
    else :
        target_pix_fmt ='p010le'if SETTINGS .encoder_type =='nvenc'and source_pix_fmt =='yuv420p10le'else source_pix_fmt 

    base_args =[]
    if SETTINGS.encoder_type == 'nvenc':
        # Start with the encoder and preset
        base_args = ['-c:v', 'av1_nvenc', '-preset', SETTINGS.nvenc_preset]
        
        # Add the -tune parameter based on the quality mode setting
        if SETTINGS.nvenc_quality_mode:
            base_args.extend(['-tune', SETTINGS.nvenc_quality_mode.lower()])

        # Add the rest of the quality and rate control parameters
        base_args.extend(['-rc', 'vbr', '-cq', str(quality_value), '-b:v', '0'])
        
        # Add advanced parameters if they exist for the final encode
        if for_final_encode and SETTINGS.nvenc_advanced_params:
            base_args.extend(SETTINGS.nvenc_advanced_params.split())

    elif SETTINGS.encoder_type == 'svt_av1':
        # Start with the basic encoder arguments
        base_args = [
            '-c:v', 'libsvtav1',
            '-preset', str(SETTINGS.svt_av1_preset),
            '-crf', str(quality_value)
        ]

        # Only add advanced parameters if they exist for the final encode
        if for_final_encode and SETTINGS.svt_av1_advanced_params:
            # Ensure the parameter string is not just empty spaces
            if SETTINGS.svt_av1_advanced_params.strip():
                 base_args.extend(['-svtav1-params', SETTINGS.svt_av1_advanced_params])

    else:
        raise ValueError(f"Unsupported encoder type in config: {SETTINGS.encoder_type}")

    if target_pix_fmt :
        base_args .extend (['-pix_fmt',target_pix_fmt ])
    base_args .extend (color_args )
    return base_args 


def format_duration (seconds :float )->str :
    """Formats seconds into a human-readable string (e.g., 1h 23m 45s)."""
    if seconds <0 :
        seconds =0 
    seconds =int (seconds )
    mins ,secs =divmod (seconds ,60 )
    hours ,mins =divmod (mins ,60 )
    if hours >0 :
        return f"{hours }h {mins }m {secs }s"
    if mins >0 :
        return f"{mins }m {secs }s"
    return f"{secs }s"


def get_file_size_info (filepath :str )->tuple [int ,float ]:
    """Returns the file size in bytes and megabytes."""
    if os .path .exists (filepath ):
        size_bytes =os .path .getsize (filepath )
        return size_bytes ,size_bytes /(1024 *1024 )
    return 0 ,0.0 


def classify_resolution (width :Optional [int ],height :Optional [int ])->str :
    if not width or not height :
        return "1080p"
    if width >=3840 :
        return "4K"
    if width >=1920 :
        return "1080p"
    if width >=1280 :
        return "720p"
    return "SD"


def calculate_vmaf (encoded_path :str ,reference_path :str ,n_threads :int ,progress_callback =None )->float :
    """Calculates VMAF score."""
    if SETTINGS .vmaf_targeting_mode !='percentile':

        options ={'log_path':os .devnull }
        if n_threads >0 :
            options ['n_threads']=n_threads 
        option_string =':'.join ([f"{key }={value }"for key ,value in options .items ()])
        filter_string =f"[0:v]setpts=PTS-STARTPTS[dist];[1:v]setpts=PTS-STARTPTS[ref];[dist][ref]libvmaf={option_string }"
        cmd =[SETTINGS .ffmpeg_path ,'-i',encoded_path ,'-i',reference_path ,'-lavfi',filter_string ,'-f','null','-']

        try :
            result =subprocess .run (cmd ,capture_output =True ,text =True ,encoding ='utf-8',
            check =True ,env =FFMPEG_ENV ,creationflags =SUBPROCESS_FLAGS )
            vmaf_match =re .search (r'VMAF score: ([\d\.]+)',result .stderr )
            if not vmaf_match :
                vmaf_match =re .search (r'VMAF.*?:\s*([\d\.]+)',result .stderr )
            return float (vmaf_match .group (1 ))if vmaf_match else 0.0 
        except Exception as e :
            if progress_callback :
                progress_callback (f"VMAF calculation failed: {e }")
            return 0.0 


    log_path =None 
    try :
        unique_id =uuid .uuid4 ()
        log_path =f"vmaf_log_{unique_id }.json"

        options ={'log_fmt':'json','log_path':log_path }
        if n_threads >0 :
            options ['n_threads']=n_threads 
        option_string =':'.join ([f"{k }={v }"for k ,v in options .items ()])

        filter_string =f"[0:v]setpts=PTS-STARTPTS[dist];[1:v]setpts=PTS-STARTPTS[ref];[dist][ref]libvmaf={option_string }"
        cmd =[SETTINGS .ffmpeg_path ,'-i',encoded_path ,'-i',reference_path ,
        '-lavfi',filter_string ,'-f','null','-']

        result =subprocess .run (cmd ,capture_output =True ,text =True ,encoding ='utf-8',
        check =False ,env =FFMPEG_ENV ,creationflags =SUBPROCESS_FLAGS ,timeout =300 )

        if result .returncode !=0 :
            raise RuntimeError (f"FFmpeg failed with code {result .returncode }")

        if not os .path .exists (log_path ):
            raise RuntimeError (f"VMAF log file was not created")

        with open (log_path ,'r')as f :
            vmaf_data =json .load (f )

        frames =vmaf_data .get ('frames',[])
        if not frames :
            raise ValueError ("VMAF log file contains no frame data")

        frame_scores =[]
        for frame in frames :
            if 'metrics'in frame and 'vmaf'in frame ['metrics']:
                score =frame ['metrics']['vmaf']
                if isinstance (score ,(int ,float ))and 0 <=score <=100 :
                    frame_scores .append (score )

        if not frame_scores :
            raise ValueError ("No valid VMAF scores found")

        percentile_value =float (np .percentile (frame_scores ,SETTINGS .vmaf_target_percentile ))

        if progress_callback :
            progress_callback (f"VMAF P{SETTINGS .vmaf_target_percentile }: {percentile_value :.2f}")

        return percentile_value 

    except Exception as e :
        print (f"[VMAF] Error during calculation: {e }")
        return 0.0 
    finally :
        if log_path and os .path .exists (log_path ):
            try :
                os .unlink (log_path )
            except :
                pass 


def calculate_score_with_FFVShip (reference_path :str ,distorted_path :str ,metric :str )->float :
    """Calculates a quality score using the FFVShip CLI tool."""
    if not os .path .exists (SETTINGS .ffvship_path ):
        raise FileNotFoundError (f"FFVShip not found at: {SETTINGS .ffvship_path }")

    metric_map ={
    "ssimulacra2":"SSIMULACRA2",
    "butteraugli":"Butteraugli"
    }
    ffvq_metric =metric_map .get (metric )
    if not ffvq_metric :
        raise ValueError (f"Unsupported metric for FFVShip: {metric }")

    cmd =[
    SETTINGS .ffvship_path ,
    '--source',reference_path ,
    '--encoded',distorted_path ,
    '-m',ffvq_metric 
    ]

    try :
        result =subprocess .run (cmd ,capture_output =True ,text =True ,check =False ,
        timeout =300 ,creationflags =SUBPROCESS_FLAGS )

        if result .returncode !=0 :
            raise RuntimeError (f"FFVShip failed with code {result .returncode }: {result .stderr }")

        score_pattern =re .compile (r"Average\s*:\s*([\d\.]+)")
        match =score_pattern .search (result .stdout )

        if not match :
            raise ValueError (f"Could not parse score from FFVShip output")

        return float (match .group (1 ))

    except subprocess .TimeoutExpired :
        raise RuntimeError ("FFVShip timed out during quality calculation.")
    except Exception as e :
        raise RuntimeError (f"Error running FFVShip: {e }")


def get_tier1_samples (input_path :str ,log_callback ,task_id :int ,video_duration :float )->Optional [Tuple [List [float ],Dict [str ,float ]]]:
    """Tier 1: High-speed scene detection using FFmpeg with complexity analysis."""
    try :
        threshold =getattr (SETTINGS ,'ffmpeg_scenedetect_threshold',0.4 )
        log_callback (task_id ,f"Tier 1: Using FFmpeg scene detection (threshold={threshold })...")

        cmd =[
        SETTINGS .ffmpeg_path ,
        '-i',input_path ,
        '-vf',f"select='gt(scene,{threshold })',showinfo",
        '-f','null','-'
        ]

        result =subprocess .run (cmd ,capture_output =True ,text =True ,encoding ='utf-8',
        env =FFMPEG_ENV ,timeout =300 ,creationflags =SUBPROCESS_FLAGS )

        timestamps =[]
        time_pattern =re .compile (r'pts_time:([\d\.]+)')
        for line in result .stderr .split ('\n'):
            if 'pts_time:'in line :
                match =time_pattern .search (line )
                if match :
                    timestamps .append (float (match .group (1 )))

        if len (timestamps )<SETTINGS .min_scene_changes_required :
            log_callback (task_id ,f"Tier 1: Found only {len (timestamps )} scenes, need {SETTINGS .min_scene_changes_required }.")
            return None 


        scene_durations =[]
        if len (timestamps )>1 :
            for i in range (len (timestamps )-1 ):
                scene_durations .append (timestamps [i +1 ]-timestamps [i ])
        if timestamps :
            scene_durations .append (video_duration -timestamps [-1 ])

        avg_scene_duration =sum (scene_durations )/len (scene_durations )if scene_durations else 0 
        min_scene_duration =min (scene_durations )if scene_durations else 0 
        scenes_per_minute =(len (timestamps )/video_duration )*60 if video_duration >0 else 0 
        quick_cuts =sum (1 for d in scene_durations if d <2.0 )
        quick_cut_ratio =quick_cuts /len (scene_durations )if scene_durations else 0 
        complexity_score =min (1.0 ,(scenes_per_minute /30.0 )*0.5 +quick_cut_ratio *0.5 )

        complexity_data ={
        'scene_count':len (timestamps ),
        'avg_scene_duration':avg_scene_duration ,
        'min_scene_duration':min_scene_duration ,
        'scenes_per_minute':scenes_per_minute ,
        'quick_cut_ratio':quick_cut_ratio ,
        'complexity_score':complexity_score ,
        'sampling_method':'tier1_ffmpeg'
        }

        log_callback (task_id ,f"Tier 1: Detected {len (timestamps )} scenes, complexity: {complexity_score :.2f}")


        timestamps =[ts for ts in timestamps 
        if SETTINGS .skip_start_seconds <=ts <=(video_duration -SETTINGS .skip_end_seconds )]


        if len (timestamps )>SETTINGS .num_samples :
            indices =[int (i *(len (timestamps )-1 )/(SETTINGS .num_samples -1 ))
            for i in range (SETTINGS .num_samples )]
            selected =sorted ([timestamps [i ]for i in set (indices )])
        else :
            selected =timestamps 

        log_callback (task_id ,f"Tier 1: Selected {len (selected )} samples")
        return selected ,complexity_data 

    except Exception as e :
        log_callback (task_id ,f"Tier 1 Error: {e }")
        return None 


def get_tier2_samples (input_path :str ,log_callback ,task_id :int ,metrics :dict )->Optional [Tuple [List [float ],Dict [str ,float ]]]:
    """Tier 2: SmartFrames - Temporal bucketing with keyframe complexity scoring."""
    try :
        video_duration =metrics ['video_duration_seconds']

        log_callback (task_id ,"Tier 2: Extracting keyframes...")
        cmd_keyframes =[SETTINGS .ffmpeg_path ,'-skip_frame','nokey','-i',input_path ,
        '-vf','showinfo','-f','null','-']
        result =subprocess .run (cmd_keyframes ,capture_output =True ,text =True ,encoding ='utf-8',
        env =FFMPEG_ENV ,timeout =120 ,creationflags =SUBPROCESS_FLAGS )

        all_keyframes =[]
        for line in result .stderr .split ('\n'):
            if 'pts_time:'in line :
                time_m =re .search (r'pts_time:([\d\.]+)',line )
                if time_m :
                    timestamp =float (time_m .group (1 ))
                    all_keyframes .append (timestamp )

        if not all_keyframes :
            log_callback (task_id ,"Tier 2: No keyframes found")
            return None 

        log_callback (task_id ,f"Tier 2: Found {len (all_keyframes )} keyframes")


        keyframe_scores ={}
        for i ,kf_time in enumerate (all_keyframes ):
            nearby_count =sum (1 for other_kf in all_keyframes 
            if abs (other_kf -kf_time )<=2.0 and other_kf !=kf_time )

            prev_dist =all_keyframes [i ]-all_keyframes [i -1 ]if i >0 else float ('inf')
            next_dist =all_keyframes [i +1 ]-all_keyframes [i ]if i <len (all_keyframes )-1 else float ('inf')
            min_dist =min (prev_dist ,next_dist )

            density_score =nearby_count *2.0 
            distance_score =1.0 /max (min_dist ,0.1 )if min_dist <float ('inf')else 0.0 

            keyframe_scores [kf_time ]=density_score +distance_score 


        work_start =SETTINGS .skip_start_seconds 
        work_end =video_duration -SETTINGS .skip_end_seconds 
        work_duration =work_end -work_start 

        if work_duration <SETTINGS .sample_segment_duration *SETTINGS .num_samples :
            work_start =0 
            work_end =video_duration 
            work_duration =video_duration 

        bucket_duration =work_duration /SETTINGS .num_samples 
        buckets =[(work_start +(i *bucket_duration ),work_start +((i +1 )*bucket_duration ))
        for i in range (SETTINGS .num_samples )]


        selected_timestamps =[]
        for bucket_start ,bucket_end in buckets :
            bucket_keyframes =[(kf ,keyframe_scores .get (kf ,0 ))
            for kf in all_keyframes if bucket_start <=kf <bucket_end ]

            if bucket_keyframes :
                bucket_keyframes .sort (key =lambda x :x [1 ],reverse =True )
                selected_timestamps .append (bucket_keyframes [0 ][0 ])
            else :
                center_time =(bucket_start +bucket_end )/2 
                if center_time +SETTINGS .sample_segment_duration <=video_duration :
                    selected_timestamps .append (center_time )
                else :
                    adjusted_time =max (bucket_start ,video_duration -SETTINGS .sample_segment_duration )
                    selected_timestamps .append (adjusted_time )

        selected_timestamps .sort ()


        keyframe_intervals =[all_keyframes [i ]-all_keyframes [i -1 ]
        for i in range (1 ,len (all_keyframes ))]

        avg_interval =sum (keyframe_intervals )/len (keyframe_intervals )if keyframe_intervals else 10.0 
        min_interval =min (keyframe_intervals )if keyframe_intervals else 0 

        keyframes_per_minute =(len (all_keyframes )/video_duration )*60 if video_duration >0 else 0 
        quick_cuts =sum (1 for interval in keyframe_intervals if interval <2.0 )
        quick_cut_ratio =quick_cuts /len (keyframe_intervals )if keyframe_intervals else 0 

        complexity_score =min (1.0 ,(keyframes_per_minute /30.0 )*0.5 +quick_cut_ratio *0.5 )

        complexity_data ={
        'scene_count':len (all_keyframes ),
        'avg_scene_duration':avg_interval ,
        'min_scene_duration':min_interval ,
        'scenes_per_minute':keyframes_per_minute ,
        'quick_cut_ratio':quick_cut_ratio ,
        'complexity_score':complexity_score ,
        'sampling_method':'tier2_temporal_buckets'
        }

        log_callback (task_id ,f"Tier 2: Selected {len (selected_timestamps )} samples, complexity: {complexity_score :.2f}")
        return selected_timestamps ,complexity_data 

    except Exception as e :
        log_callback (task_id ,f"Tier 2 Error: {e }")
        return None 


def get_tier3_samples (metrics :dict ,log_callback ,task_id :int )->List [float ]:
    """Tier 3: Gets sample points based on even intervals."""
    video_duration =metrics ['video_duration_seconds']
    use_full_duration =False 

    effective_duration =video_duration -SETTINGS .skip_start_seconds -SETTINGS .skip_end_seconds 
    required_time =SETTINGS .num_samples *SETTINGS .sample_segment_duration 

    if effective_duration <required_time :
        log_callback (task_id ,"Tier 3: Not enough time for samples after skipping. Using full duration.")
        use_full_duration =True 

    if use_full_duration :
        start_offset =0 
        duration_to_use =video_duration 
    else :
        start_offset =SETTINGS .skip_start_seconds 
        duration_to_use =effective_duration 

    interval =((duration_to_use -required_time )/(SETTINGS .num_samples -1 )
    if SETTINGS .num_samples >1 and duration_to_use >required_time else 0 )

    return [start_offset +(i *(SETTINGS .sample_segment_duration +interval ))
    for i in range (SETTINGS .num_samples )]


def get_final_sample_points (input_path :str ,task_id :int ,log_callback ,
metrics :dict )->Tuple [List [float ],Optional [Dict [str ,float ]]]:
    """Get sample points using the configured method."""
    if hasattr (threading .current_thread (),'stop_event')and threading .current_thread ().stop_event .is_set ():
        return [],None 

    samples =None 
    complexity_data =None 
    video_duration =metrics ['video_duration_seconds']


    if SETTINGS .sampling_method =='tier1':
        log_callback (task_id ,"Attempting FFmpeg Scene Detection - Tier 1...")
        result =get_tier1_samples (input_path ,log_callback ,task_id ,video_duration )
        if result :
            samples ,complexity_data =result 
            return samples ,complexity_data 
        log_callback (task_id ,"Tier 1 failed, falling back to Tier 2...")


    if SETTINGS .sampling_method in ['tier1','tier2']:
        log_callback (task_id ,"Attempting SmartFrames Analysis - Tier 2...")
        result =get_tier2_samples (input_path ,log_callback ,task_id ,metrics )
        if result :
            samples ,complexity_data =result 
            return samples ,complexity_data 
        log_callback (task_id ,"Tier 2 failed, falling back to Tier 3...")


    log_callback (task_id ,"Using Time Intervals - Tier 3...")
    samples =get_tier3_samples (metrics ,log_callback ,task_id )

    complexity_data ={
    'scene_count':0 ,
    'avg_scene_duration':0 ,
    'min_scene_duration':0 ,
    'scenes_per_minute':0 ,
    'quick_cut_ratio':0 ,
    'complexity_score':0.5 ,
    'sampling_method':'tier3'
    }

    return samples ,complexity_data 


def create_master_sample_in_memory (input_path :str ,sample_points :List [float ],task_id :int ,
log_callback ,video_stream :Dict )->tuple [bytes ,float ]:
    """Create master sample in memory."""
    if hasattr (threading .current_thread (),'stop_event')and threading .current_thread ().stop_event .is_set ():
        raise RuntimeError ("Process aborted")

    start_time =time .time ()
    log_callback (task_id ,f"Creating master sample from {len (sample_points )} segments...")

    color_args =build_color_args (video_stream )
    pix_fmt_arg =['-pix_fmt',video_stream ['pix_fmt']]if video_stream .get ('pix_fmt')else []


    filter_parts =[]
    for i ,start_s in enumerate (sample_points ):
        filter_parts .append (f"[0:v]trim=start={start_s }:duration={SETTINGS .sample_segment_duration },setpts=PTS-STARTPTS[seg{i }]")

    concat_filter ="".join ([f"[seg{i }]"for i in range (len (sample_points ))])+f"concat=n={len (sample_points )}:v=1:a=0[out]"
    filter_complex =";".join (filter_parts )+";"+concat_filter 


    """Temporary solution for 10bit 444p source video"""
    if SETTINGS .master_sample_encoder =='nvenc' and '444p10' in pix_fmt_arg:
        encoder_cmd =['-c:v','hevc_nvenc','-preset','lossless','-qp','0']
    elif SETTINGS .master_sample_encoder =='nvenc':
        encoder_cmd = ['-c:v','h264_nvenc','-preset','lossless','-qp','0']
    elif SETTINGS .master_sample_encoder =='raw':
        encoder_cmd =['-c:v','rawvideo']
    else :
        encoder_cmd =['-c:v','libx264','-preset','ultrafast','-crf','0']

    cmd =[
    SETTINGS .ffmpeg_path ,'-i',input_path ,
    '-filter_complex',filter_complex ,
    '-map','[out]',
    *pix_fmt_arg ,*color_args ,*encoder_cmd ,
    '-f','matroska','pipe:1'
    ]

    result =subprocess .run (cmd ,capture_output =True ,env =FFMPEG_ENV ,timeout =300 ,creationflags =SUBPROCESS_FLAGS )
    if result .returncode !=0 :
        raise RuntimeError (f"Master sample creation failed: {result .stderr .decode ('utf-8',errors ='ignore')}")

    log_callback (task_id ,"Master sample created in RAM")
    return result .stdout ,time .time ()-start_time 


def encode_sample_in_memory (reference_data :bytes ,cq :int ,video_stream :Dict )->bytes :
    """Encodes a sample in memory."""
    encoder_args =build_encoder_args (cq ,video_stream ,for_final_encode =False )

    input_temp =tempfile .NamedTemporaryFile (delete =False ,suffix ='.mkv',prefix ='enc_input_')
    output_temp =tempfile .NamedTemporaryFile (delete =False ,suffix ='.mkv',prefix ='enc_output_')

    try :
        input_temp .write (reference_data )
        input_temp .flush ()
        input_temp .close ()
        output_temp .close ()

        if not os .path .exists (input_temp .name )or os .path .getsize (input_temp .name )==0 :
            raise RuntimeError ("Failed to create valid input temporary file")

        cmd =[
        SETTINGS .ffmpeg_path ,
        '-v','error',
        '-i',input_temp .name ,
        *encoder_args ,
        '-movflags','+faststart',
        '-avoid_negative_ts','make_zero',
        '-fflags','+genpts',
        '-y',output_temp .name 
        ]

        result =subprocess .run (cmd ,capture_output =True ,text =True ,env =FFMPEG_ENV ,
        timeout =500 ,encoding ='utf-8',creationflags =SUBPROCESS_FLAGS )

        if result .returncode !=0 :
            error_msg =result .stderr [-1000 :]if result .stderr else "Unknown encoding error"
            raise RuntimeError (f"Encoding failed: {error_msg }")

        if not os .path .exists (output_temp .name ):
            raise RuntimeError ("Encoded output file was not created")

        output_size =os .path .getsize (output_temp .name )
        if output_size ==0 :
            raise RuntimeError ("Encoded output file is empty")

        with open (output_temp .name ,'rb')as f :
            encoded_data =f .read ()

        if len (encoded_data )==0 :
            raise RuntimeError ("No data read from encoded file")

        return encoded_data 

    except subprocess .TimeoutExpired :
        raise RuntimeError ("Encoding timed out")
    except Exception as e :
        raise RuntimeError (f"Encoding failed: {str (e )}")
    finally :
        for temp_file in [input_temp .name ,output_temp .name ]:
            try :
                if os .path .exists (temp_file ):
                    os .unlink (temp_file )
            except (OSError ,PermissionError ):
                pass 


def run_quality_test_optimized (cq ,master_sample_path ,video_stream ,log_callback ,task_id ,
n_threads :int ,progress_tracker =None ,subtask_id =None ,
input_path =None ,sample_timestamps =None ,complexity_score =None ,
file_features =None, encoder_features=None ):
    """Run quality test with caching and feature storage."""
    test_start_time =time .time ()

    try:
        # --- NEW: Create a dictionary of the current encoder settings for the cache key ---
        encoder_settings = {}
        if SETTINGS.encoder_type == 'nvenc':
            encoder_settings['preset'] = SETTINGS.nvenc_preset
            encoder_settings['quality_mode'] = SETTINGS.nvenc_quality_mode
        elif SETTINGS.encoder_type == 'svt_av1':
            encoder_settings['preset'] = str(SETTINGS.svt_av1_preset)

        if SETTINGS.enable_quality_cache and input_path and sample_timestamps:
            mode = SETTINGS.quality_metric_mode
            metric_subtype = get_vmaf_subtype(SETTINGS) if mode == 'vmaf' else mode
            
            # --- MODIFIED: Pass the encoder_settings to the cache function ---
            cached_score = database_manager.get_quality_cache(
                input_path, sample_timestamps, cq, mode, complexity_score, 
                metric_subtype=metric_subtype,
                encoder_settings=encoder_settings
            )
            if cached_score is not None:
                log_callback(task_id, f"  -> CQ {cq}: {mode.upper()}: {cached_score:.2f} [cached]")
                if progress_tracker and subtask_id is not None:
                    progress_tracker.update_subtask_progress(subtask_id, 100.0)
                return cq, cached_score, 0.0


        with open (master_sample_path ,'rb')as f :
            master_sample_data =f .read ()

        encoded_sample_data =encode_sample_in_memory (master_sample_data ,cq ,video_stream )

        with tempfile .NamedTemporaryFile (delete =False ,suffix ='.mkv')as enc_file :
            enc_file .write (encoded_sample_data )
            enc_path =enc_file .name 


        score_to_return =0.0 
        mode =SETTINGS .quality_metric_mode 

        try :
            if mode =='vmaf':
                vmaf_score =calculate_vmaf (enc_path ,master_sample_path ,n_threads =n_threads )
                score_to_return =vmaf_score 

                if SETTINGS .vmaf_targeting_mode =='percentile':
                    log_msg =f"  -> CQ {cq }: VMAF (P{SETTINGS .vmaf_target_percentile }): {vmaf_score :.2f}"
                else :
                    log_msg =f"  -> CQ {cq }: VMAF: {vmaf_score :.2f}"
                log_callback (task_id ,log_msg )

            elif mode =='ssimulacra2':
                s2_score =calculate_score_with_FFVShip (master_sample_path ,enc_path ,'ssimulacra2')
                score_to_return =s2_score 
                log_callback (task_id ,f"  -> CQ {cq }: SSIMULACRA2: {s2_score :.2f}")

            elif mode =='butteraugli':
                b_score =calculate_score_with_FFVShip (master_sample_path ,enc_path ,'butteraugli')
                score_to_return =b_score 
                log_callback (task_id ,f"  -> CQ {cq }: BUTTERAUGLI: {b_score :.2f}")


            if SETTINGS.enable_quality_cache and input_path and sample_timestamps:
                metric_subtype = get_vmaf_subtype(SETTINGS) if mode == 'vmaf' else mode
                
                # --- MODIFIED: Combine all features and pass them for storage ---
                all_features = {}
                if file_features: all_features.update(file_features)
                if encoder_features: all_features.update(encoder_features)

                database_manager.set_quality_cache(
                    input_path, sample_timestamps, cq, mode, score_to_return,
                    complexity_score, all_features,
                    metric_subtype=metric_subtype,
                    encoder_settings=encoder_settings
                )

        finally :
            os .unlink (enc_path )

        if progress_tracker and subtask_id is not None :
            progress_tracker .update_subtask_progress (subtask_id ,100.0 )

        del master_sample_data 
        del encoded_sample_data 
        gc .collect ()

        return cq ,score_to_return ,time .time ()-test_start_time 

    except Exception as e :
        log_callback (task_id ,f"Quality test failed for CQ {cq }: {e }")
        if progress_tracker and subtask_id is not None :
            progress_tracker .update_subtask_progress (subtask_id ,100.0 )
        return cq ,0.0 ,time .time ()-test_start_time




def find_best_cq_optimized(input_path: str, task_id: int, log_callback, metrics: dict,
                          worker_progress_objects: Dict, lock: threading.Lock, video_stream: Dict,
                          quality_model: QualityModel, complexity_data: dict,
                          file_features: dict, error_analyzer: PredictionErrorAnalyzer = None) -> tuple[Optional[int], dict]:
    """ML-enhanced CQ search with confidence-based strategy and intelligent fallbacks."""
    timings = {}
    master_sample_temp_file = None
    
    if not video_stream:
        log_callback(task_id, "ERROR: Could not find video stream for CQ search.")
        return None, timings
    
    # Get encoder features
    encoder_features = FeatureExtractor.get_encoder_features(SETTINGS)
    
    mode = SETTINGS.quality_metric_mode
    target_score = SETTINGS.target_score
    higher_is_better = (mode != 'butteraugli')
    tolerance = (SETTINGS.quality_tolerance_percent / 100.0) * target_score
    mode_display = f"VMAF P{SETTINGS.vmaf_target_percentile}" if mode == 'vmaf' and SETTINGS.vmaf_targeting_mode == 'percentile' else mode.upper()
    
    log_callback(task_id, f"INFO: 🎯 Target: {mode_display} {target_score:.2f} ±{tolerance:.3f} [{target_score - tolerance:.2f} - {target_score + tolerance:.2f}]")
    
    cq_search_path = {}
    tested_cqs = {}
    
    try:
        if hasattr(threading.current_thread(), 'stop_event') and threading.current_thread().stop_event.is_set():
            log_callback(task_id, "Search aborted by user")
            return None, timings
        
        # Create master sample
        sample_points_list = complexity_data.get('sample_points')
        if not sample_points_list:
            log_callback(task_id, "ERROR: Failed to determine sample points from analysis phase.")
            return None, timings
        
        master_sample_data, timings['sample_creation_time'] = create_master_sample_in_memory(
            input_path, sample_points_list, task_id, log_callback, video_stream
        )
        
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mkv', prefix=f'master_sample_worker{task_id}_') as ref_file:
            ref_file.write(master_sample_data)
            master_sample_temp_file = ref_file.name
        
        log_callback(task_id, f"Master sample created ({len(master_sample_data)/(1024*1024):.1f} MB)")
        del master_sample_data
        gc.collect()
        
        thread_count = SETTINGS.cpu_threads if SETTINGS.cpu_threads > 0 else 0
        
        # Helper function for running tests
        def run_test(cq):
            if cq in tested_cqs:
                return cq, tested_cqs[cq], 0
            _, score, test_time = run_quality_test_optimized(
                cq, master_sample_temp_file, video_stream,
                log_callback, task_id, thread_count, None, None,
                input_path, sample_points_list, complexity_data.get('complexity_score'),
                file_features=file_features,
                encoder_features=encoder_features
            )
            tested_cqs[cq] = score
            cq_search_path[cq] = score
            return cq, score, test_time
        
        total_work_time = 0
        
        # ML PREDICTION WITH CONFIDENCE
        if quality_model and quality_model.is_trained:
            log_callback(task_id, f"INFO: Using ML model ({quality_model.metric_name.upper()}) with confidence assessment...")
            
            # Get prediction with confidence
            predicted_cq, predicted_score, confidence, all_predictions = quality_model.predict_with_confidence(
                file_features, encoder_features, target_score, tolerance,
                SETTINGS.cq_search_min, SETTINGS.cq_search_max
            )
            
            if predicted_cq is not None:
                # Apply learned correction if available
                if error_analyzer:
                    correction = error_analyzer.get_correction_factor(file_features)
                    if abs(correction) >= 0.5:
                        corrected_cq = int(round(predicted_cq + correction))
                        corrected_cq = max(SETTINGS.cq_search_min, min(SETTINGS.cq_search_max, corrected_cq))
                        log_callback(task_id, f"  -> Applying learned correction: {predicted_cq} → {corrected_cq}")
                        predicted_cq = corrected_cq
                
                log_callback(task_id, f"  -> ML predicts CQ {predicted_cq} (confidence: {confidence.upper()})")
                
                # CONFIDENCE-BASED STRATEGY
                
                if confidence == 'high':
                    # HIGH CONFIDENCE: Aggressive strategy
                    log_callback(task_id, "  -> High confidence: Testing predicted+1 first (aggressive)")
                    
                    # Test predicted+1 first (based on observed bias)
                    if predicted_cq + 1 <= SETTINGS.cq_search_max:
                        _, score_plus1, t = run_test(predicted_cq + 1)
                        total_work_time += t
                        
                        if target_score - tolerance <= score_plus1 <= target_score + tolerance:
                            # Try +2 for even better compression
                            if predicted_cq + 2 <= SETTINGS.cq_search_max:
                                _, score_plus2, t = run_test(predicted_cq + 2)
                                total_work_time += t
                                
                                if target_score - tolerance <= score_plus2 <= target_score + tolerance:
                                    final_cq = predicted_cq + 2
                                    final_score = score_plus2
                                else:
                                    final_cq = predicted_cq + 1
                                    final_score = score_plus1
                            else:
                                final_cq = predicted_cq + 1
                                final_score = score_plus1
                            
                            log_callback(task_id, f"SUCCESS: ✅ High confidence success! CQ {final_cq} = {final_score:.2f}")
                            timings.update({
                                'best_cq': final_cq,
                                'final_score': final_score,
                                'cq_search_path': json.dumps(cq_search_path),
                                'quality_search_time': total_work_time,
                                'search_iterations': len(tested_cqs),
                                'ml_confidence': confidence,
                                'ml_accelerated': True
                            })
                            return final_cq, timings
                    
                    # Fallback to predicted if +1 failed
                    _, score_pred, t = run_test(predicted_cq)
                    total_work_time += t
                    
                    if target_score - tolerance <= score_pred <= target_score + tolerance:
                        final_cq = predicted_cq
                        final_score = score_pred
                        log_callback(task_id, f"SUCCESS: ✅ CQ {final_cq} = {final_score:.2f}")
                        timings.update({
                            'best_cq': final_cq,
                            'final_score': final_score,
                            'cq_search_path': json.dumps(cq_search_path),
                            'quality_search_time': total_work_time,
                            'search_iterations': len(tested_cqs),
                            'ml_confidence': confidence,
                            'ml_accelerated': True
                        })
                        return final_cq, timings
                    
                    # High confidence but still wrong - narrow search
                    log_callback(task_id, "  -> High confidence miss, using narrow search")
                    confidence = 'medium'  # Downgrade confidence
                
                if confidence == 'medium':
                    # MEDIUM CONFIDENCE: Balanced strategy
                    log_callback(task_id, "  -> Medium confidence: Testing predicted, then boundary")
                    
                    # Test predicted value
                    _, score_pred, t = run_test(predicted_cq)
                    total_work_time += t
                    
                    if target_score - tolerance <= score_pred <= target_score + tolerance:
                        # In range, try to optimize
                        best_cq = predicted_cq
                        best_score = score_pred
                        
                        # Test +1 for better compression
                        if predicted_cq + 1 <= SETTINGS.cq_search_max:
                            _, score_plus1, t = run_test(predicted_cq + 1)
                            total_work_time += t
                            
                            if target_score - tolerance <= score_plus1 <= target_score + tolerance:
                                best_cq = predicted_cq + 1
                                best_score = score_plus1
                                
                                # Try +2
                                if predicted_cq + 2 <= SETTINGS.cq_search_max:
                                    _, score_plus2, t = run_test(predicted_cq + 2)
                                    total_work_time += t
                                    
                                    if target_score - tolerance <= score_plus2 <= target_score + tolerance:
                                        best_cq = predicted_cq + 2
                                        best_score = score_plus2
                        
                        log_callback(task_id, f"SUCCESS: ✅ CQ {best_cq} = {best_score:.2f}")
                        timings.update({
                            'best_cq': best_cq,
                            'final_score': best_score,
                            'cq_search_path': json.dumps(cq_search_path),
                            'quality_search_time': total_work_time,
                            'search_iterations': len(tested_cqs),
                            'ml_confidence': confidence,
                            'ml_accelerated': True
                        })
                        return best_cq, timings
                    
                    # Not in range, do targeted search
                    if score_pred > target_score + tolerance:
                        # Quality too high, search higher CQ
                        search_min = predicted_cq + 1
                        search_max = min(predicted_cq + 8, SETTINGS.cq_search_max)
                    else:
                        # Quality too low, search lower CQ
                        search_min = max(predicted_cq - 8, SETTINGS.cq_search_min)
                        search_max = predicted_cq - 1
                    
                    log_callback(task_id, f"  -> Medium confidence miss, searching {search_min}-{search_max}")
                
                elif confidence == 'low':
                    # LOW CONFIDENCE: Conservative strategy
                    log_callback(task_id, "  -> Low confidence: Sampling multiple points")
                    
                    # Sample 3 points to triangulate
                    sample_points = [
                        max(predicted_cq - 2, SETTINGS.cq_search_min),
                        predicted_cq,
                        min(predicted_cq + 2, SETTINGS.cq_search_max)
                    ]
                    
                    for cq in sample_points:
                        _, score, t = run_test(cq)
                        total_work_time += t
                    
                    # Find best range based on samples
                    passing_cqs = [cq for cq, score in tested_cqs.items()
                                  if target_score - tolerance <= score <= target_score + tolerance]
                    
                    if passing_cqs:
                        final_cq = max(passing_cqs)
                        final_score = tested_cqs[final_cq]
                        log_callback(task_id, f"SUCCESS: ✅ Found CQ {final_cq} = {final_score:.2f}")
                        timings.update({
                            'best_cq': final_cq,
                            'final_score': final_score,
                            'cq_search_path': json.dumps(cq_search_path),
                            'quality_search_time': total_work_time,
                            'search_iterations': len(tested_cqs),
                            'ml_confidence': confidence,
                            'ml_accelerated': True
                        })
                        return final_cq, timings
                    
                    # Determine search range from samples
                    scores_sorted = sorted(tested_cqs.items(), key=lambda x: abs(x[1] - target_score))
                    best_sample = scores_sorted[0][0]
                    search_min = max(best_sample - 3, SETTINGS.cq_search_min)
                    search_max = min(best_sample + 3, SETTINGS.cq_search_max)
                    
                    log_callback(task_id, f"  -> Low confidence, searching {search_min}-{search_max}")
                
                else:
                    # No confidence/failed prediction
                    search_min = SETTINGS.cq_search_min
                    search_max = SETTINGS.cq_search_max
                
                # Store error for learning if we got here
                if error_analyzer and predicted_cq in tested_cqs:
                    # Find actual optimal from tested values
                    passing = [(cq, s) for cq, s in tested_cqs.items() 
                              if target_score - tolerance <= s <= target_score + tolerance]
                    if passing:
                        optimal_cq = max(passing, key=lambda x: x[0])[0]
                        error_analyzer.analyze_error(
                            file_features, predicted_cq, optimal_cq,
                            predicted_score, tested_cqs[predicted_cq]
                        )
            else:
                search_min = SETTINGS.cq_search_min
                search_max = SETTINGS.cq_search_max
        else:
            search_min = SETTINGS.cq_search_min
            search_max = SETTINGS.cq_search_max
        
        # FALLBACK: Original interpolation search for remaining range
        if 'best_cq' not in locals():
            log_callback(task_id, f"Starting interpolation search in range {search_min}-{search_max}...")
            
            # Test boundaries if not already tested
            for cq in [search_min, search_max]:
                if cq not in tested_cqs:
                    _, score, t = run_test(cq)
                    total_work_time += t
            
            # Check if target is achievable
            min_score = tested_cqs.get(search_min, 0)
            max_score = tested_cqs.get(search_max, 0)
            
            if (higher_is_better and min_score < target_score - tolerance) or \
               (not higher_is_better and min_score > target_score + tolerance):
                log_callback(task_id, f"ERROR: ❌ Target unachievable. Best: {min_score:.2f} at CQ {search_min}")
                timings.update({
                    'search_iterations': len(tested_cqs),
                    'cq_search_path': json.dumps(cq_search_path),
                    'quality_search_time': total_work_time,
                    'best_cq': None,
                    'final_score': min_score
                })
                return None, timings
            
            # Interpolation search
            low_cq, high_cq = search_min, search_max
            low_score, high_score = min_score, max_score
            
            for iteration in range(SETTINGS.max_iterations):
                if high_cq - low_cq <= 1:
                    break
                
                if abs(high_score - low_score) > 0.001:
                    interpolated = low_cq + (target_score - low_score) * (high_cq - low_cq) / (high_score - low_score)
                else:
                    interpolated = (low_cq + high_cq) / 2
                
                next_cq = round(interpolated)
                next_cq = max(low_cq + 1, min(high_cq - 1, next_cq))
                
                if next_cq in tested_cqs:
                    next_cq = round((low_cq + high_cq) / 2)
                    if next_cq in tested_cqs:
                        break
                
                log_callback(task_id, f"Iteration {iteration + 1}: Testing CQ {next_cq}")
                _, score, t = run_test(next_cq)
                total_work_time += t
                
                if (higher_is_better and score >= target_score - tolerance) or \
                   (not higher_is_better and score <= target_score + tolerance):
                    low_cq, low_score = next_cq, score
                else:
                    high_cq, high_score = next_cq, score
            
            # Find best CQ
            passing_cqs = [cq for cq, score in tested_cqs.items()
                          if target_score - tolerance <= score <= target_score + tolerance]
            
            if not passing_cqs:
                best_cq_found = max(tested_cqs, key=tested_cqs.get) if higher_is_better else min(tested_cqs, key=tested_cqs.get)
                best_score_found = tested_cqs[best_cq_found]
                log_callback(task_id, f"FAILURE: No CQ found within tolerance.")
                timings.update({
                    'search_iterations': len(tested_cqs),
                    'cq_search_path': json.dumps(cq_search_path),
                    'quality_search_time': total_work_time,
                    'best_cq': None,
                    'final_score': best_score_found
                })
                return None, timings
            
            final_cq = max(passing_cqs)
            final_score = tested_cqs[final_cq]
        
        timings.update({
            'search_iterations': len(tested_cqs),
            'cq_search_path': json.dumps(cq_search_path),
            'quality_search_time': total_work_time,
            'best_cq': final_cq,
            'final_score': final_score,
            'ml_accelerated': quality_model is not None and quality_model.is_trained
        })
        
        log_callback(task_id, f"SUCCESS: ✅ Found optimal CQ {final_cq} with score {final_score:.2f}")
        return final_cq, timings
        
    except Exception as e:
        import traceback
        log_callback(task_id, f"FATAL error in CQ search: {type(e).__name__}: {str(e)}\n{traceback.format_exc()}")
        timings['quality_search_time'] = timings.get('quality_search_time', 0)
        return None, timings
    finally:
        if master_sample_temp_file and os.path.exists(master_sample_temp_file):
            try:
                os.unlink(master_sample_temp_file)
            except Exception as e:
                print(f"Warning: Could not delete master sample file: {e}")
        with lock:
            if task_id in worker_progress_objects:
                worker_progress_objects[task_id].clear()
        gc.collect()





def get_system_info ()->Dict :
    """Gather system information for logging."""
    system_info ={
    'batch_id':str (uuid .uuid4 ()),
    'cpu_name':'Unknown',
    'gpu_name':'Unknown',
    'ffmpeg_version':'Unknown'
    }

    # ADD: Define startupinfo once for all calls in this function
    startupinfo = None
    if os.name == 'nt':
        startupinfo = subprocess.STARTUPINFO()
        startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
        startupinfo.wShowWindow = subprocess.SW_HIDE

    try :
        system_info ['cpu_name']=platform .processor ()or 'Unknown'

        if os .name =='nt':
            try :
                result =subprocess .run (
                ['wmic','cpu','get','name'],
                capture_output =True ,text =True ,timeout =5 ,creationflags =SUBPROCESS_FLAGS,
                startupinfo=startupinfo # MODIFIED: Add startupinfo
                )
                if result .returncode ==0 and result .stdout :
                    lines =result .stdout .strip ().split ('\n')
                    if len (lines )>1 :
                        cpu_name =lines [1 ].strip ()
                        if cpu_name and cpu_name !='Name':
                            system_info ['cpu_name']=cpu_name 
            except :
                pass 
        elif platform .system ()=="Linux":
            try :
                with open ('/proc/cpuinfo','r')as f :
                    for line in f :
                        if line .startswith ('model name'):
                            system_info ['cpu_name']=line .split (':')[1 ].strip ()
                            break 
            except :
                pass 
    except Exception as e :
        print (f"Warning: Could not get CPU info: {e }")

    try :
        if SETTINGS and SETTINGS .encoder_type =='nvenc':
            try :
                result =subprocess .run (
                ['nvidia-smi','--query-gpu=name','--format=csv,noheader'],
                capture_output =True ,text =True ,timeout =5 ,creationflags =SUBPROCESS_FLAGS,
                startupinfo=startupinfo # MODIFIED: Add startupinfo
                )
                if result .returncode ==0 and result .stdout :
                    gpu_name =result .stdout .strip ().split ('\n')[0 ]
                    if gpu_name :
                        system_info ['gpu_name']=gpu_name 
            except FileNotFoundError :
                system_info ['gpu_name']='NVIDIA GPU (nvidia-smi not found)'
            except :
                system_info ['gpu_name']='NVIDIA GPU (detection failed)'
        else :
            system_info ['gpu_name']='Not using GPU'
    except Exception as e :
        print (f"Warning: Could not get GPU info: {e }")

    try :
        if SETTINGS and os .path .exists (SETTINGS .ffmpeg_path ):
            result =subprocess .run (
            [SETTINGS .ffmpeg_path ,'-version'],
            capture_output =True ,text =True ,timeout =5 ,creationflags =SUBPROCESS_FLAGS,
            startupinfo=startupinfo # MODIFIED: Add startupinfo
            )
            if result .returncode ==0 and result .stdout :
                lines =result .stdout .split ('\n')
                if lines :
                    version_line =lines [0 ]
                    if 'version'in version_line .lower ():
                        version_match =re .search (r'version\s+([^\s]+)',version_line ,re .IGNORECASE )
                        if version_match :
                            system_info ['ffmpeg_version']=version_match .group (1 )
    except Exception as e :
        print (f"Warning: Could not get FFmpeg version: {e }")

    return system_info






def log_enhanced_results (input_path :str ,output_path :str ,cq_used :int ,duration :float ,
status :str ,metrics :Dict ,timings :dict ,video_stream :Optional [Dict ]):
    """Logs results to both the legacy text file and the database."""
    timestamp =time .strftime ("%Y-%m-%d %H:%M:%S")
    input_size =metrics .get ('input_size_bytes',0 )
    video_duration =metrics .get ('video_duration_seconds',0 )

    output_size_bytes ,output_size_mb =get_file_size_info (output_path )

    source_bitrate =(input_size *8 )/(video_duration *1000 )if video_duration >0 else 0 
    output_bitrate =(output_size_bytes *8 )/(video_duration *1000 )if video_duration >0 and output_size_bytes >0 else 0 
    space_saved =(input_size -output_size_bytes )/(1024 *1024 )if output_size_bytes >0 else 0 
    compression_ratio =input_size /output_size_bytes if output_size_bytes >0 else 0 
    processing_speed =video_duration /duration if duration >0 else 0 


    log_entry =(
    f"--- Encoding Log ({timestamp }) ---\n"
    f"File: {Path (input_path ).name }\nStatus: {status }\n"
    f"Encoder: {SETTINGS .encoder_type if SETTINGS else 'unknown'}\nCQ/CRF Value: {cq_used }\n"
    f"Input Size: {input_size /(1024 *1024 ):.2f} MB\nOutput Size: {output_size_mb :.2f} MB\n"
    f"Space Saved: {space_saved :.2f} MB\nCompression Ratio: {compression_ratio :.2f}x\n"
    f"Source Bitrate: {source_bitrate :.2f} kb/s\nOutput Bitrate: {output_bitrate :.2f} kb/s\n"
    f"Processing Time: {format_duration (duration )}\nProcessing Speed: {processing_speed :.2f}x\n"
    f"----------------------------------------\n\n"
    )

    try :
        if SETTINGS and SETTINGS .encoding_log_path :
            log_dir =os .path .dirname (SETTINGS .encoding_log_path )
            if log_dir and not os .path .exists (log_dir ):
                os .makedirs (log_dir ,exist_ok =True )

            with open (SETTINGS .encoding_log_path ,'a',encoding ='utf-8')as f :
                f .write (log_entry )
    except Exception as e :
        print (f"Failed to write to log file: {e }")


    if SETTINGS and SETTINGS .enable_performance_log and "Success"in status :
        try :

            framerate =metrics .get ('frame_rate',0 )
            if duration >0 and framerate >0 and video_duration >0 :
                total_frames =video_duration *framerate 
                metrics ['final_encode_fps']=total_frames /duration 
            else :
                metrics ['final_encode_fps']=0 

            metrics ['size_after_mb']=output_size_mb 
            metrics ['final_bitrate_kbps']=output_bitrate 


            predicted_duration =metrics .get ('initial_predicted_duration',0 )
            if predicted_duration >0 and duration >0 :
                metrics ['prediction_error_ratio']=(duration -predicted_duration )/predicted_duration 
                error_percent =abs (metrics ['prediction_error_ratio']*100 )
                if error_percent >50 :
                    print (f"Large prediction error: {error_percent :.1f}% for {Path (input_path ).name }")
            else :
                metrics ['prediction_error_ratio']=0 


            if timings :
                for key in ['best_cq','cq_search_path','search_iterations',
                'quality_search_time','final_score','sample_creation_time']:
                    if key in timings :
                        metrics [key ]=timings [key ]


            if database_manager:
                database_manager.log_performance(metrics)
                
                # Record performance prediction error for learning
                if 'initial_predicted_duration' in metrics and metrics['initial_predicted_duration'] > 0:
                    if 'final_encode_fps' in metrics and metrics['final_encode_fps'] > 0:
                        # Get the features that were used for prediction
                        file_features = {k: metrics.get(k, 0) for k in [
                            'resolution_pixels', 'source_bitrate_kbps', 'bitrate_per_pixel',
                            'complexity_score', 'scenes_per_minute', 'frame_rate', 'is_10bit', 'is_hdr'
                        ]}
                        encoder_features = {
                            'is_nvenc': 1 if metrics.get('encoder_type') == 'nvenc' else 0,
                            'preset_num': metrics.get('preset_num', 5)
                        }
                        
                        # Calculate what FPS was predicted vs actual
                        video_duration = metrics.get('video_duration_seconds', 0)
                        frame_rate = metrics.get('frame_rate', 30.0)
                        if video_duration > 0 and frame_rate > 0:
                            total_frames = video_duration * frame_rate
                            predicted_fps = total_frames / metrics['initial_predicted_duration']
                            actual_fps = metrics['final_encode_fps']
                            
                            # Find the performance error analyzer in the parent app
                            current_thread = threading.current_thread()
                            if hasattr(current_thread, 'perf_error_analyzer'):
                                current_thread.perf_error_analyzer.record_error(
                                    file_features, encoder_features, 
                                    predicted_fps, actual_fps
                                )

        except Exception as e :
            print (f"ERROR: Failed to log performance to database: {e }")








class ThreadSafeProgressCallback :
    """Thread-safe wrapper for progress callbacks."""
    def __init__ (self ,callback_queue :queue .Queue ,worker_id :int ):
        self .callback_queue =callback_queue 
        self .worker_id =worker_id 
        self .lock =threading .Lock ()
        self .last_update_time =0 
        self .update_interval =0.5 

    def update (self ,stage :str ,progress :float ):
        """Send progress update if enough time has passed."""
        current_time =time .time ()
        with self .lock :
            if current_time -self .last_update_time >=self .update_interval :
                try :
                    self .callback_queue .put ({
                    'type':'worker_progress',
                    'worker_id':self .worker_id ,
                    'stage':stage ,
                    'progress':progress 
                    },block =False )
                    self .last_update_time =current_time 
                except queue .Full :
                    pass 

    def log (self ,message :str ):
        """Send log message."""
        try :
            self .callback_queue .put ({
            'type':'worker_log',
            'worker_id':self .worker_id ,
            'message':message 
            },block =False )
        except queue .Full :
            pass 



def rename_skipped_file(filepath: str, task_id: int, log_callback, reason_suffix: str = None):
    """Rename a skipped file with the configured suffix."""
    if not SETTINGS.rename_skipped_files:
        return

    try:
        p_input = Path(filepath)
        if not p_input.exists():
            return

        # Use the specific suffix if provided, otherwise use the default
        suffix_to_use = reason_suffix if reason_suffix is not None else SETTINGS.skipped_file_suffix

        if p_input.stem.endswith(suffix_to_use):
            return

        new_name = f"{p_input.stem}{suffix_to_use}{p_input.suffix}"
        new_path = p_input.parent / new_name

        if new_path.exists():
            log_callback(task_id, f"WARNING: ⚠️ Skipping rename: '{new_path.name}' already exists.")
            return

        p_input.rename(new_path)
        log_callback(task_id, f"Renamed source to '{new_path.name}'")
        print(f"[Rename] {filepath} -> {new_path.name}")

    except Exception as e:
        log_callback(task_id, f"Error renaming file {filepath}: {e}")
        print(f"[Rename] Error: {e}")


def read_stderr_non_blocking (process ,timeout =30 ):
    """Read stderr without blocking."""
    if os .name =='nt':
        q =queue .Queue ()

        def enqueue_output (out ,q ):
            try :
                for line in iter (out .readline ,''):
                    q .put (line )
            finally :
                out .close ()
                q .put (None )

        t =threading .Thread (target =enqueue_output ,args =(process .stderr ,q ),daemon =True )
        t .start ()

        while process .poll ()is None or not q .empty ():
            try :
                line =q .get (timeout =timeout )
                if line is None :
                    break 
                yield line 
            except queue .Empty :
                yield None 
    else :
        while True :
            ready ,_ ,_ =select .select ([process .stderr ],[],[],timeout )
            if not ready :
                if process .poll ()is not None :
                    break 
                yield None 
                continue 

            line =process .stderr .readline ()
            if not line :
                break 
            yield line 


def log_error (filename :str ,reason :str ):
    """Logs an error message to a separate errors.log file."""
    try :
        with open ("errors.log","a",encoding ="utf-8")as f :
            timestamp =datetime .now ().strftime ("%Y-%m-%d %H:%M:%S")
            f .write (f"[{timestamp }] {filename }: {reason }\n")
    except Exception as e :
        print (f"CRITICAL: Failed to write to error log: {e }")


def run_encode (input_path :str ,cq_value :int ,task_id :int ,batch_state :Dict ,lock :threading .Lock ,
metrics :dict ,log_callback ,timings :dict ,callback_queue :queue .Queue ,video_stream :Dict )->tuple[bool, str]:
    """Runs the final encode with robust monitoring and returns a specific reason on failure."""
    start_time =time .time ()
    p_input =Path (input_path )

    output_dir =Path (SETTINGS .output_directory )if SETTINGS .output_directory and SETTINGS .output_directory .strip ()else p_input .parent
    output_dir .mkdir (parents =True ,exist_ok =True )

    temp_output =output_dir /f"{p_input .stem }_temp{p_input .suffix }"
    final_output =output_dir /f"{p_input .stem }{SETTINGS .output_suffix }{p_input .suffix }"

    if temp_output .exists ():
        try :
            temp_output .unlink ()
        except OSError :
            pass

    if not video_stream :
        log_callback (task_id ,"Could not find video stream. Cannot proceed with final encode.")
        return False, "Could not find video stream for final encode"

    encoder_args =build_encoder_args (cq_value ,video_stream ,for_final_encode =True )
    cmd =[
    SETTINGS .ffmpeg_path ,'-y','-i',str (p_input ),
    *encoder_args ,
    '-map','0:v','-map','0:a?','-map','0:s?',
    '-c:a','copy','-c:s','copy',
    '-f','matroska',str (temp_output )
    ]

    log_callback (task_id ,f"INFO: 🚀 Starting final encode with {SETTINGS .encoder_type .upper ()} (CQ/CRF: {cq_value })...")

    last_size =0
    last_activity_time =time .time ()
    last_progress_time =time .time ()
    last_reported_progress =0
    stuck_at_100_time =None
    monitor_exception =None

    MAX_IDLE_TIME =300
    MAX_STUCK_AT_100_TIME =120
    PROGRESS_TIMEOUT =600

    try :
        process =subprocess .Popen (cmd ,stderr =subprocess .PIPE ,universal_newlines =True ,bufsize =1 ,
        encoding ='utf-8',env =FFMPEG_ENV ,creationflags =SUBPROCESS_FLAGS )
        duration =metrics ['video_duration_seconds']
        time_pattern =re .compile (r"time=(\d{2}):(\d{2}):(\d{2})\.(\d{2})")

        stop_monitor =threading .Event ()

        def monitor_file_activity ():
            nonlocal last_size ,last_activity_time ,monitor_exception
            try :
                while not stop_monitor .is_set ():
                    if temp_output .exists ():
                        try :
                            current_size =temp_output .stat ().st_size
                            if current_size >last_size :
                                last_size =current_size
                                last_activity_time =time .time ()
                        except (OSError ,IOError ):
                            pass

                    idle_time =time .time ()-last_activity_time
                    if idle_time >MAX_IDLE_TIME :
                        monitor_exception =RuntimeError (f"File hasn't grown for {idle_time :.0f}s (stalled)")
                        break

                    stop_monitor .wait (5 )
            except Exception as e :
                monitor_exception =e

        monitor_thread =threading .Thread (target =monitor_file_activity ,daemon =True )
        monitor_thread .start ()

        for line in read_stderr_non_blocking (process ,timeout =30 ):
            if line is None :
                if process .poll ()is not None :
                    break
                if time .time ()-last_progress_time >PROGRESS_TIMEOUT :
                    raise RuntimeError (f"No progress updates for {PROGRESS_TIMEOUT }s")

                if stuck_at_100_time and (time .time ()-stuck_at_100_time >MAX_STUCK_AT_100_TIME ):
                    raise RuntimeError (f"Stalled at 100% for over {MAX_STUCK_AT_100_TIME }s")

                continue

            last_progress_time =time .time ()
            m =time_pattern .search (line )
            if m and duration >0 :
                h ,m_str ,s ,ms =map (int ,m .groups ())
                current_time =h *3600 +m_str *60 +s +ms /100
                progress =min (100.0 ,(current_time /duration )*100 )

                if progress >=99.9 :
                    if stuck_at_100_time is None :
                        stuck_at_100_time =time .time ()
                        log_callback (task_id ,"Reached 100%, waiting for finalization...")

                if progress >last_reported_progress :
                    callback_queue .put ({
                    'type':'worker_progress',
                    'worker_id':task_id ,
                    'stage':f'Encoding... {progress :.1f}%',
                    'progress':progress
                    })
                    last_reported_progress =progress

            if monitor_exception :
                raise monitor_exception

        stop_monitor .set ()
        monitor_thread .join (timeout =5 )

        log_callback (task_id ,"Waiting for FFmpeg to exit...")
        try :
            process .wait (timeout =60 )
        except subprocess .TimeoutExpired :
            log_callback (task_id ,"FFmpeg did not exit cleanly. Terminating...")
            process .terminate ()
            try :
                process .wait (timeout =10 )
            except subprocess .TimeoutExpired :
                log_callback (task_id ,"FFmpeg did not respond. Killing...")
                process .kill ()
                process .wait (timeout =10 )

        if process .returncode !=0 and process .returncode is not None :
            reason = f"FFmpeg returned exit code {process.returncode}"
            log_callback (task_id ,f"Encoding failed: {reason}")
            return False, reason

        if not temp_output .exists ()or temp_output .stat ().st_size <1024 :
            reason = "Encoding produced an empty or missing file"
            log_callback (task_id ,reason)
            return False, reason


        log_callback (task_id ,"Verifying output file...")
        output_info =get_media_info (str (temp_output ))
        if not (output_info and 'format'in output_info ):
            reason = "Verification failed: Could not read encoded file"
            log_callback (task_id ,reason)
            return False, reason

        output_duration =float (output_info ['format'].get ('duration',0 ))
        duration_diff =abs (output_duration -metrics ['video_duration_seconds'])
        if duration_diff >2.0 :
            reason = "Verification failed: Duration mismatch!"
            log_callback (task_id ,reason)
            return False, reason


        output_size ,input_size =temp_output .stat ().st_size ,metrics ['input_size_bytes']
        size_reduction_percent =((input_size -output_size )/input_size )*100

        if size_reduction_percent <SETTINGS .min_size_reduction_threshold :
            reason = f"Insufficient size reduction ({size_reduction_percent:.1f}%)"
            log_callback(task_id, reason)
            temp_output.unlink()
            return False, reason

        if final_output .exists ():
            try :
                final_output .unlink ()
            except OSError as e :
                log_callback (task_id ,f"Could not overwrite existing file: {e }")

        try :
            temp_output .rename (final_output )
        except OSError as e :
            reason = f"Could not rename temp file: {e}"
            log_callback (task_id ,f"FATAL: {reason}")
            return False, reason



        if SETTINGS .delete_source_file :
            p_input .unlink ()
            log_callback (task_id ,f"SUCCESS: ✅ Source deleted, saved {size_reduction_percent :.1f}%")
        else :
            log_callback (task_id ,f"SUCCESS: ✅ Source kept, saved {size_reduction_percent :.1f}%")

        with lock :
            batch_state ['deleted_source_size']=batch_state .get ('deleted_source_size',0 )+input_size
            batch_state ['encoded_output_size']=batch_state .get ('encoded_output_size',0 )+output_size

        log_enhanced_results (str (p_input ),str (final_output ),cq_value ,time .time ()-start_time ,
        "Success",metrics ,timings ,video_stream )
        return True, "Success"

    except Exception as e :
        reason = f"Encoding error: {e}"
        log_callback (task_id ,reason)
        if 'process'in locals ()and process .poll ()is None :
            process .kill ()
            process .wait ()
        return False, reason
    finally :
        if temp_output .exists ()and not final_output .exists ():
            try :
                temp_output .unlink ()
            except OSError :
                pass 


def handle_failed_file (filepath :str ,task_id :int ,lock :threading .Lock ,batch_state :Dict ,
reason :str ,metrics :dict ,video_stream :Optional [Dict ],log_callback ):
    """Handle a failed encoding."""
    metrics ['size_after_mb']=0 
    metrics ['final_bitrate_kbps']=0 
    metrics ['final_encode_fps']=0 
    metrics ['skipped_on_failure']=1 

    filename =Path (filepath ).name 
    log_error (filename ,reason )

    print (f"⚠ Skipped: {filename }")
    print (f"  Reason: {reason }")

    log_callback (task_id ,f"Skipped: {filename } - {reason }")

    log_enhanced_results (filepath ,"",0 ,0 ,f"Skipped: {reason }",metrics ,{},video_stream )

    if SETTINGS .rename_skipped_files and "stopped by user"not in reason .lower ():
        try :
            rename_skipped_file (filepath ,task_id ,log_callback )
        except Exception as e :
            log_callback (task_id ,f"Warning: Failed to rename skipped file: {e }")




def enhanced_task_worker(filepath: str, task_id: int, lock: threading.Lock, batch_state: Dict,
                         worker_progress_objects: Dict, system_info: Dict, stop_event: threading.Event,
                         callback_queue: queue.Queue, log_callback,
                         analysis_cache: dict, quality_model: QualityModel, performance_model: PerformanceModel,
                         perf_error_analyzer: PerformanceErrorAnalyzer = None):
    """Enhanced worker with ML integration."""
    task_start_time =time .time ()

    metrics ={'worker_start_timestamp':task_start_time ,'source_filename':os .path .basename (filepath )}
    metrics .update (system_info )

    feature_extractor =FeatureExtractor ()

    cached_data =analysis_cache .get (filepath )
    if not cached_data :
        reason ="Analysis data not found (aborted by user?)"
        handle_failed_file (filepath ,task_id ,lock ,batch_state ,reason ,metrics ,None ,log_callback )
        return False

    media_info =cached_data ['media_info']
    complexity_data =cached_data .get ('complexity_data',{})

    file_features =feature_extractor .extract_video_features (media_info ,complexity_data )
    encoder_features =feature_extractor .get_encoder_features (SETTINGS )
    all_features ={**file_features ,**encoder_features }

    video_stream =next ((s for s in media_info .get ('streams',[])
    if s .get ('codec_type')=='video'),None )

    try :
        if SETTINGS .enable_performance_log and database_manager :
            metrics ['file_hash']=database_manager ._get_file_hash (filepath )

        if not (media_info and 'format'in media_info and
        float (media_info .get ('format',{}).get ('duration',0 ))>0 ):
            handle_failed_file (filepath ,task_id ,lock ,batch_state ,
            "Could not extract media info",{},None ,log_callback )
            return False

        file_size ,size_mb =get_file_size_info (filepath )
        metrics ['video_duration_seconds']=float (media_info .get ('format',{}).get ('duration',0 ))

        metrics .update (file_features )
        metrics .update ({
        'input_size_bytes':file_size ,
        'size_before_mb':size_mb ,
        'container_format':media_info .get ('format',{}).get ('format_name','unknown'),
        'encoder_type':SETTINGS .encoder_type ,
        'preset':SETTINGS .nvenc_preset if SETTINGS .encoder_type =='nvenc'else str (SETTINGS .svt_av1_preset ),
        'preset_num':encoder_features ['preset_num'],
        'output_bit_depth_setting':SETTINGS .output_bit_depth ,
        'quality_metric':SETTINGS .quality_metric_mode ,
        'target_score':SETTINGS .target_score ,
        'num_samples_requested':SETTINGS .num_samples ,
        'sample_duration_s':SETTINGS .sample_segment_duration ,
        'total_sample_duration_s':SETTINGS .num_samples *SETTINGS .sample_segment_duration ,
        'sampling_method':complexity_data .get ('sampling_method','unknown')
        })

        if video_stream :
            metrics ['source_codec']=video_stream .get ('codec_name','unknown')
            metrics ['source_pixel_format']=video_stream .get ('pix_fmt','unknown')
            metrics ['resolution_key']=classify_resolution (file_features ['width'],file_features ['height'])


        if stop_event .is_set ():
            handle_failed_file (filepath ,task_id ,lock ,batch_state ,
            "Stopped by user",metrics ,video_stream ,log_callback )
            return False


        error_analyzer = PredictionErrorAnalyzer(database_manager) if database_manager else None

        optimal_cq, timings = find_best_cq_optimized(
            filepath, task_id, log_callback, metrics, worker_progress_objects,
            lock, video_stream, quality_model, complexity_data, file_features,
            error_analyzer=error_analyzer
        )

        if optimal_cq and quality_model and quality_model.is_trained and error_analyzer:
            if 'ml_confidence' in timings and database_manager:
                # Store the prediction performance for learning
                features_json = json.dumps({**file_features, **encoder_features})
                if 'ml_predicted_cq' in timings:  # Would need to add this to timings
                    database_manager.store_prediction_error(
                        metrics.get('file_hash', ''),
                        timings['ml_predicted_cq'],
                        optimal_cq,
                        timings.get('ml_predicted_score', 0),
                        timings['final_score'],
                        features_json
                    )

        metrics .update (timings )

        if optimal_cq is None :
            if SETTINGS .skip_encoding_if_target_not_reached :
                handle_failed_file (filepath ,task_id ,lock ,batch_state ,
                "Target quality not achievable",metrics ,video_stream ,log_callback )
                return False
            else :
                optimal_cq =timings .get ('best_cq')or SETTINGS .cq_search_min

        if performance_model and performance_model .is_trained and optimal_cq is not None :
            try :
                predicted_fps ,confidence =performance_model .predict_fps (all_features )

                if predicted_fps >0 and metrics .get ('video_duration_seconds',0 )>0 :
                    frame_rate =metrics .get ('frame_rate',30.0 )
                    total_frames =metrics ['video_duration_seconds']*frame_rate
                    predicted_encode_duration =total_frames /predicted_fps

                    metrics ['initial_predicted_duration']=predicted_encode_duration

                    log_callback (task_id ,f"ML predicts encoding will take ~{format_duration (predicted_encode_duration )} ({confidence } confidence)")
            except Exception as e :
                log_callback (task_id ,f"Warning: Could not generate performance prediction: {e }")
        else :
            metrics ['initial_predicted_duration']=0

        if perf_error_analyzer:
            threading.current_thread().perf_error_analyzer = perf_error_analyzer

        success, reason = run_encode(filepath, optimal_cq, task_id, batch_state, lock,
                                     metrics, log_callback, timings, callback_queue, video_stream)

        if not success:
            handle_failed_file(filepath, task_id, lock, batch_state,
                               reason, metrics, video_stream, log_callback)

    except Exception as e :
        import traceback
        log_callback (task_id ,f"ERROR: Worker exception: {e }\n{traceback .format_exc ()}")
        handle_failed_file (filepath ,task_id ,lock ,batch_state ,
        f"Exception: {e }",metrics ,video_stream ,log_callback )
        return False

    return success






SETTINGS =None 
database_manager =None 
memory_manager =None 
FFMPEG_ENV ={}


class Tooltip :
    """Creates a tooltip for a given widget that appears on hover."""
    def __init__ (self ,widget ,text ):
        self .widget =widget 
        self .text =text 
        self .tooltip_window =None 
        self .widget .bind ("<Enter>",self .show_tooltip )
        self .widget .bind ("<Leave>",self .hide_tooltip )

    def show_tooltip (self ,event =None ):
        if self .tooltip_window :
            return 
        x =self .widget .winfo_rootx ()+20 
        y =self .widget .winfo_rooty ()+20 
        self .tooltip_window =ctk .CTkToplevel (self .widget )
        self .tooltip_window .wm_overrideredirect (True )
        self .tooltip_window .wm_geometry (f"+{x }+{y }")
        label =ctk .CTkLabel (
        self .tooltip_window ,
        text =self .text ,
        justify =ctk .LEFT ,
        wraplength =400 ,
        fg_color =("#333333","#DDDDDD"),
        text_color =("#DDDDDD","#333333"),
        corner_radius =6 ,
        padx =10 ,
        pady =5 
        )
        label .pack ()

    def hide_tooltip (self ,event =None ):
        if self .tooltip_window :
            self .tooltip_window .destroy ()
        self .tooltip_window =None 


class ConfigEditor (ctk .CTkToplevel ):
    """Configuration editor window with comprehensive tooltips for all settings."""
    def __init__ (self ,parent ):
        super ().__init__ (parent )
        self .parent_app =parent 

        self .title ("Auto Target Encoder - Settings Editor")
        self .geometry ("900x800")
        self .resizable (False ,False )
        self .config_file ='config.ini'

        self .grid_columnconfigure (0 ,weight =1 )
        self .grid_rowconfigure (0 ,weight =1 )

        self .tab_view =ctk .CTkTabview (self ,width =850 ,height =720 )
        self .tab_view .grid (row =0 ,column =0 ,padx =20 ,pady =10 ,sticky ="nsew")

        self .tabs ={
        "Paths":self .tab_view .add ("Paths & Files"),
        "Encoder":self .tab_view .add ("Encoder Settings"),
        "Quality":self .tab_view .add ("Quality & Metrics"),
        "Performance":self .tab_view .add ("Performance"),
        "Sampling":self .tab_view .add ("Sampling"),
        "Filtering":self .tab_view .add ("File Filtering"),
        }

        self .setting_map ={}
        self .settings_vars ={}

        self .create_paths_tab ()
        self .create_encoder_tab ()
        self .create_quality_tab ()
        self .create_performance_tab ()
        self .create_sampling_tab ()
        self .create_filtering_tab ()

        self .bottom_frame =ctk .CTkFrame (self ,height =50 )
        self .bottom_frame .grid (row =1 ,column =0 ,padx =20 ,pady =(0 ,10 ),sticky ="ew")
        self .bottom_frame .grid_columnconfigure (0 ,weight =1 )

        self .status_label =ctk .CTkLabel (self .bottom_frame ,text ="Ready",text_color ="gray")
        self .status_label .grid (row =0 ,column =0 ,padx =10 ,pady =5 ,sticky ="w")

        button_frame =ctk .CTkFrame (self .bottom_frame )
        button_frame .grid (row =0 ,column =1 ,padx =10 ,pady =5 ,sticky ="e")

        ctk .CTkButton (button_frame ,text ="Load",command =self .load_settings ,width =80 ).pack (side ="left",padx =5 )
        ctk .CTkButton (button_frame ,text ="Save",command =self .save_settings ,width =80 ).pack (side ="left",padx =5 )
        ctk .CTkButton (button_frame ,text ="Close",command =self .destroy ,width =80 ).pack (side ="left",padx =5 )

        self .load_settings ()

    def create_widget_row (self ,tab ,label_text ,widget_type ,section ,key ,row ,
    options =None ,file_type =None ,is_folder =False ,tooltip =None ,width =350 ):
        """Creates a labeled widget row with the tooltip icon next to the label."""
        label_frame =ctk .CTkFrame (tab ,fg_color ="transparent")
        label_frame .grid (row =row ,column =0 ,padx =10 ,pady =5 ,sticky ="w")

        label =ctk .CTkLabel (label_frame ,text =label_text ,anchor ="w")
        label .pack (side ="left")

        if tooltip :
            tooltip_label =ctk .CTkLabel (label_frame ,text =" (?)",text_color ="gray",cursor ="hand2")
            tooltip_label .pack (side ="left",padx =(5 ,0 ))
            Tooltip (tooltip_label ,tooltip )

        var_key =f"{section }_{key }"
        var =ctk .StringVar ()
        self .settings_vars [var_key ]=var 
        self .setting_map [var_key ]=(section ,key )

        widget_frame =ctk .CTkFrame (tab ,fg_color ="transparent")
        widget_frame .grid (row =row ,column =1 ,padx =10 ,pady =5 ,sticky ="w")

        if widget_type =="entry":
            entry =ctk .CTkEntry (widget_frame ,textvariable =var ,width =width )
            entry .pack (side ="left")
            if file_type or is_folder :
                browse_button =ctk .CTkButton (
                widget_frame ,text ="Browse",width =80 ,
                command =lambda v =var ,ft =file_type ,fld =is_folder :self .browse (v ,ft ,fld )
                )
                browse_button .pack (side ="left",padx =10 )
        elif widget_type =="combobox":
            combobox =ctk .CTkComboBox (widget_frame ,variable =var ,values =options ,width =200 )
            combobox .pack (side ="left")
        elif widget_type =="checkbox":
            checkbox =ctk .CTkCheckBox (widget_frame ,text ="",variable =var ,onvalue ='true',offvalue ='false')
            checkbox .pack (side ="left")
        elif widget_type =="spinbox":
            entry =ctk .CTkEntry (widget_frame ,textvariable =var ,width =100 )
            entry .pack (side ="left")

        return var 

    def create_paths_tab (self ):
        tab =self .tabs ["Paths"]
        self .create_widget_row (tab ,"FFmpeg Path:","entry","Paths","ffmpeg_path",1 ,file_type ="exe",tooltip ="Required. Must point to the ffmpeg.exe executable.")
        self .create_widget_row (tab ,"FFprobe Path:","entry","Paths","ffprobe_path",2 ,file_type ="exe",tooltip ="Required. Must point to the ffprobe.exe executable.")
        self .create_widget_row (tab ,"FFVShip Path:","entry","Paths","ffvship_path",3 ,file_type ="exe",tooltip ="Optional. Path to ffvship.exe, required for SSIMULACRA2 or Butteraugli metrics.")
        self .create_widget_row (tab ,"Database Path:","entry","Paths","database_path",4 ,file_type ="db",tooltip ="Path to the SQLite database file for caching and performance logs.")
        self .create_widget_row (tab ,"Encoding Log Path:","entry","Paths","encoding_log_path",5 ,file_type ="txt",tooltip ="Path to the text file for human-readable logs of completed encodes.")
        self .create_widget_row (tab ,"Input Directory:","entry","File_Management","input_directory",7 ,is_folder =True ,tooltip ="The folder to scan for video files to encode.")
        self .create_widget_row (tab ,"Output Directory:","entry","File_Management","output_directory",8 ,is_folder =True ,tooltip ="Where to save the new encoded files.\nIf left blank, files will be saved in the same directory as their source.")
        self .create_widget_row (tab ,"Delete Source Files:","checkbox","File_Management","delete_source_file",9 ,tooltip ="WARNING: If checked, the original source file will be PERMANENTLY DELETED after a successful encode.")
        self .create_widget_row (tab ,"Output Suffix:","entry","File_Management","output_suffix",10 ,width =150 ,tooltip ="The text added to the end of a successfully encoded file's name.")
        self .create_widget_row (tab ,"Skipped File Suffix:","entry","File_Management","skipped_file_suffix",11 ,width =150 ,tooltip ="The text added to a file that was skipped.")

    def create_encoder_tab (self ):
        tab =self .tabs ["Encoder"]
        self .create_widget_row (tab ,"Encoder Type:","combobox","Encoder","encoder_type",1 ,options =["nvenc","svt_av1"],tooltip ="The main choice of encoder.\n- nvenc: Uses NVIDIA GPU. Faster but lower compression.\n- svt_av1: Uses CPU. Slower but better compression.")
        self .create_widget_row (tab ,"NVENC Preset:","entry","NVENC","nvenc_preset",3 ,tooltip ="Controls speed vs quality for NVENC.\n- p1: Fastest, lowest quality.\n- p7: Slowest, best quality.")
        self .create_widget_row (tab ,"NVENC Quality Mode:","combobox","NVENC","nvenc_quality_mode",4 ,options =["UHQ","HQ"],tooltip ="Quality algorithm for NVENC.\n- UHQ: Ultra High Quality\n- HQ: High Quality")
        self .create_widget_row (tab ,"NVENC Extra Params:","entry","NVENC_Advanced","extra_params",5 ,tooltip ="Advanced FFmpeg parameters for NVENC.")
        self .create_widget_row (tab ,"SVT-AV1 Preset:","spinbox","SVT_AV1","svt_av1_preset",7 ,tooltip ="Speed preset for SVT-AV1.\n- 0: Highest quality, slowest.\n- 13: Fastest, lowest quality.")
        self .create_widget_row (tab ,"SVT-AV1 Extra Params:","entry","SVT_AV1_Advanced","extra_params",9 ,tooltip ="Advanced FFmpeg parameters for SVT-AV1.")
        self .create_widget_row (tab ,"Output Bit Depth:","combobox","Output","output_bit_depth",11 ,options =["source","8bit","10bit"],tooltip ="Color depth of output.\n- 10bit: Recommended for AV1.\n- source: Keep original.\n- 8bit: Force 8-bit.")

    def create_quality_tab (self ):
        tab =self .tabs ["Quality"]
        self .create_widget_row (tab ,"Quality Metric Mode:","combobox","Quality_Metrics","quality_metric_mode",1 ,options =["vmaf","ssimulacra2","butteraugli"],tooltip ="Metric for judging quality.\n- VMAF: Industry standard.\n- SSIMULACRA2: Modern perceptual metric.\n- Butteraugli: Detects visual distortions.")
        self .create_widget_row (tab ,"Target Score:","spinbox","Quality_Metrics","target_score",2 ,tooltip ="Target quality score. VMAF 95-98 is often 'visually lossless'.")
        self .create_widget_row (tab ,"Tolerance (%):","spinbox","Quality_Metrics","quality_tolerance_percent",3 ,tooltip ="Acceptable margin of error as percentage of target score.")
        self .create_widget_row (tab ,"CQ/CRF Search Min:","spinbox","Quality_Metrics","cq_search_min",5 ,tooltip ="Lowest quality (highest CQ/CRF number) allowed.")
        self .create_widget_row (tab ,"CQ/CRF Search Max:","spinbox","Quality_Metrics","cq_search_max",6 ,tooltip ="Highest quality (lowest CQ/CRF number) allowed.")
        self .create_widget_row (tab ,"VMAF Targeting Mode:","combobox","Quality_Metrics","vmaf_targeting_mode",8 ,options =["average","percentile"],tooltip ="How VMAF is calculated.\n- average: Fast, standard method.\n- percentile: Ensures X% of frames meet target.")
        self .create_widget_row (tab ,"VMAF Target Percentile:","spinbox","Quality_Metrics","vmaf_target_percentile",9 ,tooltip ="When using percentile mode, which percentile to target (1-99).")
        self .create_widget_row (tab ,"ML Extra CQ Check:","checkbox","Quality_Metrics","ml_extra_cq_check",10 ,tooltip ="If an ML prediction is successful, perform one extra quality test at a higher CQ value to potentially improve compression.")

    def create_performance_tab (self ):
        tab =self .tabs ["Performance"]
        self .create_widget_row (tab ,"Max Workers:","spinbox","Performance","max_workers",1 ,tooltip ="Number of videos to encode in parallel.\nRecommendation: 1 for most systems.")
        self .create_widget_row (tab ,"CPU Threads:","spinbox","Performance","cpu_threads",2 ,tooltip ="CPU threads for encoding/analysis.\n0: Auto-detect.")
        self .create_widget_row (tab ,"Parallel VMAF Runs:","spinbox","Performance","num_parallel_vmaf_runs",3 ,tooltip ="Number of quality tests to run simultaneously.")
        self .create_widget_row (tab ,"Max Search Iterations:","spinbox","Performance","max_iterations",4 ,tooltip ="Maximum CQ search iterations before giving up.")
        self .create_widget_row (tab ,"Enable Quality Cache:","checkbox","Cache","enable_quality_cache",6 ,tooltip ="Cache quality test results for faster re-testing.")
        self .create_widget_row (tab ,"Enable Performance Log:","checkbox","Cache","enable_performance_log",7 ,tooltip ="Log performance data for ML-based ETA predictions.")

    def create_sampling_tab (self ):
        tab =self .tabs ["Sampling"]
        self .create_widget_row (tab ,"Sampling Method:","combobox","Quality_Sampling","sampling_method",1 ,options =["tier1","tier2","tier3"],tooltip ="Strategy for choosing test clips.\n- Tier 1: Scene detection.\n- Tier 2: SmartFrames analysis.\n- Tier 3: Even intervals.")
        self .create_widget_row (tab ,"Master Sample Encoder:","combobox","Quality_Sampling","master_sample_encoder",2 ,options =["software","nvenc","raw"],tooltip ="Encoder for lossless reference.\n- software: x264 lossless.\n- nvenc: GPU accelerated.\n- raw: Uncompressed.")
        self .create_widget_row (tab ,"Number of Samples:","spinbox","Quality_Sampling","num_samples",4 ,tooltip ="How many clips to test.")
        self .create_widget_row (tab ,"Sample Duration (s):","spinbox","Quality_Sampling","sample_segment_duration",5 ,tooltip ="Length of each test clip in seconds.")
        self .create_widget_row (tab ,"Skip Start (s):","spinbox","Quality_Sampling","skip_start_seconds",6 ,tooltip ="Ignore first X seconds (e.g., intros).")
        self .create_widget_row (tab ,"Skip End (s):","spinbox","Quality_Sampling","skip_end_seconds",7 ,tooltip ="Ignore last X seconds (e.g., credits).")
        self .create_widget_row (tab ,"Scene Threshold:","spinbox","Quality_Sampling","ffmpeg_scenedetect_threshold",9 ,tooltip ="Sensitivity for scene detection (0.0-1.0).")
        self .create_widget_row (tab ,"Min Scene Changes:","spinbox","Quality_Sampling","min_scene_changes_required",10 ,tooltip ="Minimum scenes needed for Tier 1.")
        self .create_widget_row (tab ,"Min Keyframes:","spinbox","Quality_Sampling","min_keyframes_required",11 ,tooltip ="Minimum keyframes needed for Tier 2.")

    def create_filtering_tab (self ):
        tab =self .tabs ["Filtering"]
        self .create_widget_row (tab ,"Min Size Reduction (%):","spinbox","File_Management","min_size_reduction_threshold",1 ,tooltip ="Only keep new file if it's this % smaller than original.")
        self .create_widget_row (tab ,"Skip if Target Unreachable:","checkbox","File_Management","skip_encoding_if_target_not_reached",2 ,tooltip ="Skip final encode if quality target can't be met.")
        self .create_widget_row (tab ,"Rename Skipped Files:","checkbox","File_Management","rename_skipped_files",3 ,tooltip ="Add suffix to skipped files to prevent re-processing.")
        self .create_widget_row (tab ,"Min Duration (s):","spinbox","File_Filtering","min_duration_seconds",5 ,tooltip ="Skip videos shorter than this.")
        self .create_widget_row (tab ,"Min File Size (MB):","spinbox","File_Filtering","min_filesize_mb",6 ,tooltip ="Skip files smaller than this.")
        self .create_widget_row (tab ,"Min 4K Bitrate (kbps):","spinbox","File_Filtering","min_bitrate_4k_kbps",8 ,tooltip ="Skip 4K videos below this bitrate.")
        self .create_widget_row (tab ,"Min 1080p Bitrate (kbps):","spinbox","File_Filtering","min_bitrate_1080p_kbps",9 ,tooltip ="Skip 1080p videos below this bitrate.")
        self .create_widget_row (tab ,"Min 720p Bitrate (kbps):","spinbox","File_Filtering","min_bitrate_720p_kbps",10 ,tooltip ="Skip 720p videos below this bitrate.")

    def browse (self ,var ,file_type ,is_folder ):
        """Handles file and folder Browse."""
        if is_folder :
            path =filedialog .askdirectory ()
        else :
            if file_type =="exe":filetypes =[("Executable files","*.exe"),("All files","*.*")]
            elif file_type =="txt":filetypes =[("Text files","*.txt"),("All files","*.*")]
            elif file_type =="db":filetypes =[("Database files","*.db"),("All files","*.*")]
            else :filetypes =[("All files","*.*")]
            path =filedialog .askopenfilename (filetypes =filetypes )

        if path :
            var .set (path .replace ("\\","/"))





    def load_settings (self ):
        """Loads settings from config.ini with graceful handling of missing values."""
        if not os .path .exists (self .config_file ):
            self .status_label .configure (text ="No config file found, using defaults",text_color ="orange")
            return 

        parser =configparser .ConfigParser ()
        parser .read (self .config_file ,encoding ='utf-8')

        for var_key ,(section ,key )in self .setting_map .items ():
            try :
                if parser .has_section (section )and parser .has_option (section ,key ):
                    value =parser .get (section ,key ).strip ()
                    self .settings_vars [var_key ].set (value )
                else :

                    if 'checkbox'in str (var_key ):
                        self .settings_vars [var_key ].set ('false')
                    elif 'spinbox'in str (var_key ):
                        self .settings_vars [var_key ].set ('0')
                    else :
                        self .settings_vars [var_key ].set ('')
            except Exception as e :
                print (f"Warning loading {section }.{key }: {e }")
                self .settings_vars [var_key ].set ('')

        self .status_label .configure (text ="Settings loaded",text_color ="green")




    def save_settings (self ):
        """Saves settings to config.ini"""
        parser =configparser .ConfigParser ()

        for section ,key in self .setting_map .values ():
            if not parser .has_section (section ):
                parser .add_section (section )

        for var_key ,var_obj in self .settings_vars .items ():
            section ,key =self .setting_map [var_key ]
            parser .set (section ,key ,var_obj .get ())

        try :
            with open (self .config_file ,'w',encoding ='utf-8')as f :
                parser .write (f )
            self .status_label .configure (text ="Settings saved!",text_color ="green")
            messagebox .showinfo ("Success","Settings saved successfully!")

            if hasattr (self .parent_app ,'refresh_dashboard'):
                self .parent_app .refresh_dashboard ()

            self .destroy ()
        except Exception as e :
            self .status_label .configure (text =f"Error: {e }",text_color ="red")
            messagebox .showerror ("Error",f"Failed to save settings: {e }")








class AnalysisProgressWindow (ctk .CTkToplevel ):
    """Progress window for file analysis phase."""
    def __init__ (self ,parent ,total_files ):
        super ().__init__ (parent )
        self .parent =parent 
        self .total_files =total_files 
        self .files_analyzed =0 
        self .cancelled =False 
        self .callback_on_cancel =None 

        self .title ("Analyzing Video Files")
        self .geometry ("500x300")
        self .resizable (False ,False )


        self .transient (parent )
        self .grab_set ()
        self .lift ()
        self .attributes ('-topmost',True )


        self .protocol ("WM_DELETE_WINDOW",self .cancel_analysis )


        main_frame =ctk .CTkFrame (self )
        main_frame .pack (fill ="both",expand =True ,padx =20 ,pady =20 )


        title_label =ctk .CTkLabel (
        main_frame ,
        text ="🔍 Analyzing Video Files",
        font =ctk .CTkFont (size =20 ,weight ="bold")
        )
        title_label .pack (pady =(0 ,20 ))


        self .progress_label =ctk .CTkLabel (
        main_frame ,
        text =f"Processing: 0 / {total_files } files",
        font =ctk .CTkFont (size =14 )
        )
        self .progress_label .pack (pady =10 )


        self .progress_bar =ctk .CTkProgressBar (main_frame ,width =400 )
        self .progress_bar .pack (pady =10 )
        self .progress_bar .set (0 )


        self .current_file_label =ctk .CTkLabel (
        main_frame ,
        text ="Initializing...",
        font =ctk .CTkFont (size =12 ),
        text_color ="gray"
        )
        self .current_file_label .pack (pady =10 )


        self .status_text =ctk .CTkTextbox (main_frame ,height =100 ,width =450 )
        self .status_text .pack (pady =10 )


        self .cancel_button =ctk .CTkButton (
        main_frame ,
        text ="Cancel",
        command =self .cancel_analysis ,
        fg_color ="red",
        hover_color ="darkred",
        width =120 
        )
        self .cancel_button .pack (pady =10 )

    def update_progress (self ,files_done ,current_file =None ):
        """Update the progress display."""
        self .files_analyzed =files_done 
        progress =files_done /self .total_files if self .total_files >0 else 0 

        self .progress_bar .set (progress )
        self .progress_label .configure (
        text =f"Processing: {files_done } / {self .total_files } files ({progress *100 :.0f}%)"
        )

        if current_file :
            filename =os .path .basename (current_file )
            self .current_file_label .configure (text =f"Analyzing: {filename }")
            self .log_message (f"Analyzing: {filename }")

    def log_message (self ,message ):
        """Add a message to the status text."""
        self .status_text .insert ("end",f"{message }\n")
        self .status_text .see ("end")

    def cancel_analysis (self ):
        if not self .cancelled :
            self .cancelled =True 

            if hasattr (self .parent ,'analysis_cancel_event'):
                self .parent .analysis_cancel_event .set ()

            if self .callback_on_cancel :
                self .callback_on_cancel ()

            self .destroy ()

    def complete_analysis (self ):
        """Called when analysis is complete."""
        self .progress_bar .set (1.0 )
        self .progress_label .configure (text =f"Analysis complete: {self .total_files } files")
        self .current_file_label .configure (text ="Starting encoding...")
        self .cancel_button .configure (state ="disabled")
        self .after (1000 ,self .destroy )







class WorkerThread (threading .Thread ):
    def __init__ (self ,file_path ,worker_id ,callback_queue ,batch_state ,parent_app ):
        super ().__init__ (daemon =True )
        self .file_path =file_path 
        self .worker_id =worker_id 
        self .callback_queue =callback_queue 
        self .batch_state =batch_state 
        self .parent_app =parent_app 
        self .stop_event =threading .Event ()
        self .aborted =False 

    def run(self):
        """Run the encoding task with the correctly selected ML model."""
        try:
            def log_to_gui(task_id, message):
                if hasattr(self, 'aborted') and self.aborted:
                    return
                self.callback_queue.put({
                    'type': 'worker_log', 'worker_id': self.worker_id, 'message': message
                })

            # --- Correct Model Selection Logic ---
            quality_model_to_use = None
            if hasattr(self.parent_app, 'quality_models'):
                encoder_type = SETTINGS.encoder_type.lower()
                metric_name = SETTINGS.quality_metric_mode.lower()
                
                model_key = f"{encoder_type}_{metric_name}"
                
                if metric_name == 'vmaf':
                    vmaf_subtype = get_vmaf_subtype(SETTINGS)
                    model_key = f"{model_key}_{vmaf_subtype}"

                quality_model_to_use = self.parent_app.quality_models.get(model_key)

                if quality_model_to_use:
                    print(f"Worker {self.worker_id}: Selected model '{model_key}' for the task.")
                else:
                    print(f"Worker {self.worker_id}: WARNING - No trained model found for key '{model_key}'. ML acceleration will be disabled.")
            # --- End of Model Selection Logic ---

            success = enhanced_task_worker(
                filepath=self.file_path,
                task_id=self.worker_id,
                lock=threading.Lock(),
                batch_state=self.batch_state,
                worker_progress_objects={},
                system_info=get_system_info(),
                stop_event=self.stop_event,
                callback_queue=self.callback_queue,
                log_callback=log_to_gui,
                analysis_cache=self.parent_app.analysis_cache,
                quality_model=quality_model_to_use,
                performance_model=self.parent_app.performance_model,
                perf_error_analyzer=self.parent_app.perf_error_analyzer if hasattr(self.parent_app, 'perf_error_analyzer') else None
            )

            if not self.stop_event.is_set():
                if success:
                    self.callback_queue.put({
                        'type': 'worker_complete', 'worker_id': self.worker_id, 'success': True,
                        'file_path': self.file_path, 'message': 'Encoding completed successfully'
                    })
                else:
                    self.callback_queue.put({
                        'type': 'worker_skipped', 'worker_id': self.worker_id,
                        'file_path': self.file_path, 'reason': 'File was skipped or failed validation.'
                    })

        except Exception as e:
            if not self.stop_event.is_set():
                import traceback
                error_str = f"{str(e)}\n{traceback.format_exc()}"
                self.callback_queue.put({
                    'type': 'worker_error', 'worker_id': self.worker_id,
                    'file_path': self.file_path, 'error': error_str
                })

    def stop (self ):
        """Signals the worker thread to stop processing."""
        self .stop_event .set ()


class EncodingDashboard (ctk .CTkFrame ):
    """Dashboard with ML-enhanced ETA predictions."""
    def __init__ (self ,parent ,parent_app =None ):
        super ().__init__ (parent )
        self .parent_app =parent_app 
        self .worker_panels ={}
        self .total_files =0 
        self .start_time =None 
        self .completed_files =0 
        self .initial_eta_seconds =None 
        self .countdown_start_time =None 
        self .eta_timer_job =None 
        self .setup_ui ()

    def _create_settings_summary (self ,parent ):
        """Creates the settings summary panel."""
        container =ctk .CTkFrame (parent )
        container .pack (fill ="x",padx =10 ,pady =10 )
        container .grid_columnconfigure ((0 ,1 ,2 ),weight =1 )


        col1 =ctk .CTkFrame (container )
        col1 .grid (row =0 ,column =0 ,padx =5 ,pady =5 ,sticky ="nsew")
        ctk .CTkLabel (col1 ,text ="Encoder & Quality",font =ctk .CTkFont (weight ="bold")).pack (anchor ="w",padx =10 ,pady =2 )

        encoder_str =f"{SETTINGS .encoder_type .upper ()} ({SETTINGS .nvenc_preset }, {SETTINGS .nvenc_quality_mode })"if SETTINGS .encoder_type =='nvenc'else f"SVT-AV1 (Preset {SETTINGS .svt_av1_preset })"
        vmaf_tolerance =(SETTINGS .quality_tolerance_percent /100.0 )*SETTINGS .target_score 

        metric_names ={
        'vmaf':'VMAF',
        'ssimulacra2':'Ssimulacra2',
        'butteraugli':'Butteraugli'
        }
        metric_display =metric_names .get (SETTINGS .quality_metric_mode ,SETTINGS .quality_metric_mode .upper ())
        if SETTINGS .quality_metric_mode =='vmaf'and SETTINGS .vmaf_targeting_mode =='percentile':
            metric_display =f"VMAF P{SETTINGS .vmaf_target_percentile }"
        target_score_str =f"{SETTINGS .target_score :.1f} (+/-{vmaf_tolerance :.2f})"

        self ._create_setting_label (col1 ,"Final Encoder:",encoder_str )
        self ._create_setting_label (col1 ,f"{metric_display } Target:",target_score_str )
        self ._create_setting_label (col1 ,"CQ/CRF Range:",f"{SETTINGS .cq_search_min } - {SETTINGS .cq_search_max }")
        self ._create_setting_label (col1 ,"Output Bit Depth:",f"{SETTINGS .output_bit_depth }")


        col2 =ctk .CTkFrame (container )
        col2 .grid (row =0 ,column =1 ,padx =5 ,pady =5 ,sticky ="nsew")
        ctk .CTkLabel (col2 ,text ="Performance & Caching",font =ctk .CTkFont (weight ="bold")).pack (anchor ="w",padx =10 ,pady =2 )

        caching_str =f"Quality {'On'if SETTINGS .enable_quality_cache else 'Off'}, Performance {'On'if SETTINGS .enable_performance_log else 'Off'}"

        tier_names ={
        "tier1":"Tier 1 - Scene Detection",
        "tier2":"Tier 2 - SmartFrames",
        "tier3":"Tier 3 - Time Intervals"
        }
        sampling_str =f"{tier_names .get (SETTINGS .sampling_method ,'Unknown')} ({SETTINGS .num_samples }x{SETTINGS .sample_segment_duration }s)"
        self ._create_setting_label (col2 ,"Sampling Method:",sampling_str )
        self ._create_setting_label (col2 ,"Max Search Iterations:",f"{SETTINGS .max_iterations }")
        self ._create_setting_label (col2 ,"Workers:",f"{SETTINGS .max_workers }")
        self ._create_setting_label (col2 ,"Caching:",caching_str )


        col3 =ctk .CTkFrame (container )
        col3 .grid (row =0 ,column =2 ,padx =5 ,pady =5 ,sticky ="nsew")
        ctk .CTkLabel (col3 ,text ="File Management",font =ctk .CTkFont (weight ="bold")).pack (anchor ="w",padx =10 ,pady =2 )

        source_action_str ="Delete after encoding"if SETTINGS .delete_source_file else f"Keep and save with '{SETTINGS .output_suffix }' suffix"
        failure_action_str ="Skip if target not achievable"if SETTINGS .skip_encoding_if_target_not_reached else "Encode regardless"

        self ._create_setting_label (col3 ,"Source Files:",source_action_str )
        self ._create_setting_label (col3 ,"Success Threshold:",f"Save > {SETTINGS .min_size_reduction_threshold }%")
        self ._create_setting_label (col3 ,"On Failure:",failure_action_str )

        active_filters = []
        if SETTINGS.min_duration_seconds > 0:
            active_filters.append(f"Min Duration: {SETTINGS.min_duration_seconds}s")
        if SETTINGS.min_filesize_mb > 0:
            active_filters.append(f"Min Size: {SETTINGS.min_filesize_mb}MB")

        bitrate_filters = []
        if SETTINGS.min_bitrate_720p_kbps > 0:
            bitrate_filters.append(f"720p: {SETTINGS.min_bitrate_720p_kbps}")
        if SETTINGS.min_bitrate_1080p_kbps > 0:
            bitrate_filters.append(f"1080p: {SETTINGS.min_bitrate_1080p_kbps}")
        if SETTINGS.min_bitrate_4k_kbps > 0:
            bitrate_filters.append(f"4k: {SETTINGS.min_bitrate_4k_kbps}")

        if bitrate_filters:
            active_filters.append(f"Min Bitrate kbps: {', '.join(bitrate_filters)}")

        # Only create the label if any filters are active, otherwise show "None"
        filtering_str = ", ".join(active_filters) if active_filters else "None"
        self._create_setting_label(col3, "Filtering:", filtering_str)


    def _create_setting_label (self ,parent ,title ,value ):
        """Helper to create a title-value pair."""
        frame =ctk .CTkFrame (parent ,fg_color ="transparent")
        frame .pack (fill ="x",padx =10 ,pady =2 )
        title_label =ctk .CTkLabel (frame ,text =title ,font =ctk .CTkFont (size =14 ,weight ="bold"),width =130 ,anchor ="w")
        title_label .pack (side ="left",padx =(0 ,5 ))
        value_label =ctk .CTkLabel (frame ,text =value ,font =ctk .CTkFont (size =14 ),anchor ="w",wraplength =220 )
        value_label .pack (side ="left",expand =True ,fill ="x")

    def format_time (self ,seconds ):
        """Formats seconds into readable format."""
        if seconds <0 :seconds =0 
        seconds =int (seconds )
        mins ,secs =divmod (seconds ,60 )
        hours ,mins =divmod (mins ,60 )
        if hours >0 :return f"{hours }h {mins }m {secs }s"
        if mins >0 :return f"{mins }m {secs }s"
        return f"{secs }s"

    def reset (self ):
        """Resets the dashboard."""
        self .start_time =None 
        self .completed_files =0 
        self .overall_progress .set (0 )
        self .eta_label .configure (text ="ETA: Preparing analysis...",text_color ="gray")
        self .stop_countdown ()
        self .initial_eta_seconds =None 
        self .countdown_start_time =None 

        for worker_id ,panel_data in self .worker_panels .items ():
            panel_data ["filename_label"].configure (text =f"--- Worker {worker_id }: Idle ---")
            panel_data ["progress"].set (0 )
            log_box =panel_data ["log_box"]
            log_box .configure (state ="normal")
            log_box .delete ("1.0","end")
            log_box .configure (state ="disabled")


    def calculate_ml_based_eta (self ):
        """Calculate ETA using ML predictions."""
        if not (self .parent_app and hasattr (self .parent_app ,'video_files')):
            return 

        total_files =len (self .parent_app .video_files )
        if total_files ==0 :
            return 


        if not hasattr (self .parent_app ,'processed_files'):
            self .parent_app .processed_files =set ()

        remaining_files =[f for f in self .parent_app .video_files 
        if f not in self .parent_app .processed_files ]

        if not remaining_files :
            self .eta_label .configure (text ="ETA: Complete!",text_color ="green")
            return 

        total_eta =0 
        feature_extractor =FeatureExtractor ()
        ml_predictions_used =0 

        for file_path in remaining_files :
            try :

                cached_data =self .parent_app .analysis_cache .get (file_path )
                if cached_data :
                    media_info =cached_data ['media_info']
                    complexity_data =cached_data .get ('complexity_data',{})
                else :
                    media_info =get_media_info (file_path )
                    complexity_data ={}

                if not media_info :
                    total_eta +=300 
                    continue 


                file_features =feature_extractor .extract_video_features (media_info ,complexity_data )
                encoder_features =feature_extractor .get_encoder_features (SETTINGS )


                duration =file_features .get ('duration_seconds',0 )
                frame_rate =file_features .get ('frame_rate',30.0 )

                if duration <=0 :
                    total_eta +=300 
                    continue 


                if self .parent_app .performance_model and self .parent_app .performance_model .is_trained :
                    predicted_fps ,confidence =self .parent_app .performance_model .predict_fps (
                    file_features ,encoder_features 
                    )

                    if predicted_fps >0 :
                        total_frames =duration *frame_rate 
                        encode_time =total_frames /predicted_fps 
                        ml_predictions_used +=1 


                        sample_time =min (30 ,duration *0.1 )
                        search_time =60 if duration >60 else 30 


                        if confidence =='high':
                            file_eta =sample_time +search_time +encode_time 
                        elif confidence =='medium':
                            file_eta =(sample_time +search_time +encode_time )*1.1 
                        else :
                            file_eta =(sample_time +search_time +encode_time )*1.3 
                    else :

                        if SETTINGS .encoder_type =='nvenc':
                            file_eta =30 +(duration *0.5 )
                        else :
                            file_eta =60 +(duration *2.0 )
                else :

                    if SETTINGS .encoder_type =='nvenc':
                        file_eta =30 +(duration *0.5 )
                    else :
                        file_eta =60 +(duration *2.0 )

                total_eta +=file_eta 

            except Exception as e :
                print (f"Error calculating ETA for file: {e }")
                total_eta +=300 


        if SETTINGS .max_workers >1 and len (remaining_files )>SETTINGS .max_workers :

            total_eta =total_eta /(SETTINGS .max_workers *0.8 )

        if total_eta >0 :
            self .initial_eta_seconds =total_eta 
            self .countdown_start_time =time .time ()
            self .stop_countdown ()
            self ._update_countdown ()


            if ml_predictions_used >0 :
                print (f"ETA calculated using ML predictions for {ml_predictions_used }/{len (remaining_files )} files")


    def _update_countdown (self ):
        """Update ETA countdown."""
        if self .initial_eta_seconds is None or self .countdown_start_time is None :
            return 

        elapsed_time =time .time ()-self .countdown_start_time 
        remaining_seconds =self .initial_eta_seconds -elapsed_time 

        if remaining_seconds >0 :
            formatted_time =self .format_time (remaining_seconds )
            self .eta_label .configure (text =f"ETA: ~{formatted_time } remaining",text_color ="white")
        else :
            over_time =abs (remaining_seconds )
            formatted_time =self .format_time (over_time )
            self .eta_label .configure (text =f"ETA: +{formatted_time } over schedule",text_color ="#E74C3C")

        self .eta_timer_job =self .after (1000 ,self ._update_countdown )

    def stop_countdown (self ):
        """Stop the ETA countdown timer."""
        if hasattr (self ,'eta_timer_job')and self .eta_timer_job :
            self .after_cancel (self .eta_timer_job )
            self .eta_timer_job =None 

    def setup_ui (self ):
        """Build dashboard UI."""
        main_stack =ctk .CTkFrame (self ,fg_color ="transparent")
        main_stack .pack (fill ="both",expand =True )

        self ._create_settings_summary (main_stack )

        summary_frame =ctk .CTkFrame (main_stack )
        summary_frame .pack (fill ="x",padx =10 ,pady =5 )
        ctk .CTkLabel (summary_frame ,text ="Encoding Summary",font =ctk .CTkFont (weight ="bold")).pack ()
        self .overall_progress =ctk .CTkProgressBar (summary_frame )
        self .overall_progress .pack (fill ="x",padx =10 ,pady =(5 ,10 ))
        self .overall_progress .set (0 )

        self .eta_label =ctk .CTkLabel (summary_frame ,text ="ETA: Ready to start",font =ctk .CTkFont (size =14 ),text_color ="gray")
        self .eta_label .pack (pady =5 )

        self .workers_content_frame =ctk .CTkFrame (main_stack ,fg_color ="transparent")
        self .workers_content_frame .pack (fill ="both",expand =True ,padx =5 ,pady =5 )

        max_workers =SETTINGS .max_workers 
        num_cols =2 if max_workers >1 else 1 
        for i in range (num_cols ):
            self .workers_content_frame .grid_columnconfigure (i ,weight =1 )

        for i in range (max_workers ):
            worker_id =i +1 
            row ,col =divmod (i ,num_cols )
            self .workers_content_frame .grid_rowconfigure (row ,weight =1 )

            panel =ctk .CTkFrame (self .workers_content_frame ,border_width =1 ,border_color ="gray30")
            panel .grid (row =row ,column =col ,padx =5 ,pady =5 ,sticky ="nsew")
            panel .grid_rowconfigure (2 ,weight =1 )
            panel .grid_columnconfigure (0 ,weight =1 )

            filename_label =ctk .CTkLabel (panel ,text =f"--- Worker {worker_id }: Idle ---",font =ctk .CTkFont (weight ="bold"),anchor ="w")
            filename_label .grid (row =0 ,column =0 ,padx =10 ,pady =5 ,sticky ="ew")

            progress =ctk .CTkProgressBar (panel )
            progress .grid (row =1 ,column =0 ,padx =10 ,pady =5 ,sticky ="ew")
            progress .set (0 )

            log_box =ctk .CTkTextbox (panel ,activate_scrollbars =True ,font =ctk .CTkFont (size =12 ))
            log_box .grid (row =2 ,column =0 ,padx =10 ,pady =5 ,sticky ="nsew")

            log_box .tag_config ("success",foreground ="#2ECC71")
            log_box .tag_config ("info",foreground ="#3498DB")
            log_box .tag_config ("error",foreground ="#E74C3C")
            log_box .tag_config ("warning",foreground ="#F39C12")

            log_box .configure (state ="disabled")

            self .worker_panels [worker_id ]={"filename_label":filename_label ,"progress":progress ,"log_box":log_box ,"busy":False }

    def log_message (self ,message ,level ="info"):
        """Log a message."""
        timestamp =datetime .now ().strftime ("%H:%M:%S")
        prefix ={"error":"❌","warning":"⚠️","success":"✅"}.get (level ,"ℹ️")

        if hasattr (self ,'log_text'):
            self .log_text .insert ("end",f"[{timestamp }] {prefix } {message }\n")
            self .log_text .see ("end")

    def activate_worker (self ,worker_id ,filename ):
        """Activate a worker panel."""
        if worker_id in self .worker_panels :
            panel =self .worker_panels [worker_id ]
            panel ["filename_label"].configure (text =f"--- Worker {worker_id }: {filename } ---")
            panel ["progress"].set (0 )
            log_box =panel ["log_box"]
            log_box .configure (state ="normal")
            log_box .delete ("1.0","end")
            log_box .configure (state ="disabled")

    def deactivate_worker (self ,worker_id ,message ):
        """Deactivate a worker panel."""
        if worker_id in self .worker_panels :
            panel =self .worker_panels [worker_id ]
            panel ["filename_label"].configure (text =f"--- Worker {worker_id }: {message } ---")
            if "Success"in message or "✅"in message :
                panel ["progress"].set (1 )

    def log_worker_message (self ,worker_id ,message ):
        """Log a message to a worker's log box."""
        if worker_id in self .worker_panels :
            log_box =self .worker_panels [worker_id ]["log_box"]
            log_box .configure (state ="normal")

            tag =None 
            if message .startswith ("SUCCESS:"):
                tag ="success"
                message =message [8 :]
            elif message .startswith ("INFO:"):
                tag ="info"
                message =message [5 :]
            elif message .startswith ("ERROR:"):
                tag ="error"
                message =message [6 :]
            elif message .startswith ("WARNING:"):
                tag ="warning"
                message =message [8 :]

            if tag :
                log_box .insert ("end",f"{message }\n",tag )
            else :
                log_box .insert ("end",f"{message }\n")

            log_box .see ("end")
            log_box .configure (state ="disabled")

    def update_worker_progress (self ,worker_id ,progress ):
        """Update worker progress bar."""
        if worker_id in self .worker_panels :
            self .worker_panels [worker_id ]["progress"].set (progress /100 )

    def update_stats (self ,completed ,total ,elapsed_time ):
        """Update overall statistics."""
        self .total_files =total 
        self .completed_files =completed 

        if total >0 :
            progress =completed /total 
            self .overall_progress .set (progress )

            if completed <total :


                if hasattr (self ,'parent_app')and self .parent_app :

                    self .parent_app .calculate_ml_based_eta ()

            else :
                self .stop_countdown ()
                self .eta_label .configure (text ="ETA: Complete!",text_color ="green")






class AnalysisProgressDialog (ctk .CTkToplevel ):
    """Real-time analysis progress dialog that stays unfrozen."""

    def __init__ (self ,parent ,total_files ,callback_on_complete ,callback_on_cancel ):
        super ().__init__ (parent )
        self .parent_app =parent 
        self .total_files =total_files 
        self .callback_on_complete =callback_on_complete 
        self .callback_on_cancel =callback_on_cancel 
        self .completed_count =0 
        self .failed_count =0 
        self .analysis_results ={}
        self .cancelled =False 

        self .title ("File Analysis Progress")
        self .geometry ("800x600")
        self .resizable (True ,True )
        self .grab_set ()


        self .transient (parent )
        self .lift ()
        self .focus ()

        self .setup_ui ()
        self .protocol ("WM_DELETE_WINDOW",self .on_cancel )

    def setup_ui (self ):

        header_frame =ctk .CTkFrame (self )
        header_frame .pack (fill ="x",padx =10 ,pady =10 )

        title_label =ctk .CTkLabel (
        header_frame ,
        text ="🔍 Analyzing Video Files",
        font =ctk .CTkFont (size =18 ,weight ="bold")
        )
        title_label .pack (side ="left",padx =10 ,pady =10 )


        self .cancel_button =ctk .CTkButton (
        header_frame ,
        text ="Cancel",
        command =self .on_cancel ,
        fg_color ="red",
        hover_color ="darkred",
        width =100 
        )
        self .cancel_button .pack (side ="right",padx =10 ,pady =10 )


        progress_frame =ctk .CTkFrame (self )
        progress_frame .pack (fill ="x",padx =10 ,pady =5 )


        self .progress_label =ctk .CTkLabel (
        progress_frame ,
        text =f"Analyzing files... 0/{self .total_files } completed",
        font =ctk .CTkFont (size =14 ,weight ="bold")
        )
        self .progress_label .pack (pady =5 )

        self .progress_bar =ctk .CTkProgressBar (progress_frame )
        self .progress_bar .pack (fill ="x",padx =20 ,pady =5 )
        self .progress_bar .set (0 )


        status_frame =ctk .CTkFrame (progress_frame ,fg_color ="transparent")
        status_frame .pack (fill ="x",padx =20 ,pady =5 )

        self .completed_label =ctk .CTkLabel (status_frame ,text ="✅ Completed: 0",text_color ="green")
        self .completed_label .pack (side ="left",padx =10 )

        self .failed_label =ctk .CTkLabel (status_frame ,text ="❌ Failed: 0",text_color ="red")
        self .failed_label .pack (side ="left",padx =10 )

        self .remaining_label =ctk .CTkLabel (status_frame ,text =f"⏳ Remaining: {self .total_files }",text_color ="orange")
        self .remaining_label .pack (side ="left",padx =10 )


        list_frame =ctk .CTkFrame (self )
        list_frame .pack (fill ="both",expand =True ,padx =10 ,pady =5 )

        ctk .CTkLabel (
        list_frame ,
        text ="File Analysis Status:",
        font =ctk .CTkFont (size =12 ,weight ="bold")
        ).pack (anchor ="w",padx =10 ,pady =5 )


        self .status_text =ctk .CTkTextbox (
        list_frame ,
        font =ctk .CTkFont (family ="Consolas",size =11 ),
        activate_scrollbars =True 
        )
        self .status_text .pack (fill ="both",expand =True ,padx =10 ,pady =5 )


        self .status_text .tag_config ("completed",foreground ="#2ECC71")
        self .status_text .tag_config ("failed",foreground ="#E74C3C")
        self .status_text .tag_config ("analyzing",foreground ="#3498DB")

    def update_file_status (self ,filename ,status ,details =""):
        """Update status of a specific file."""
        if self .cancelled :
            return 

        timestamp =time .strftime ("%H:%M:%S")

        if status =="analyzing":
            message =f"[{timestamp }] 🔄 Analyzing: {filename }\n"
            tag ="analyzing"
        elif status =="completed":
            self .completed_count +=1 
            message =f"[{timestamp }] ✅ Completed: {filename }\n"
            tag ="completed"
        elif status =="failed":
            self .failed_count +=1 
            error_detail =f" ({details })"if details else ""
            message =f"[{timestamp }] ❌ Failed: {filename }{error_detail }\n"
            tag ="failed"


        self .status_text .insert ("end",message ,tag )
        self .status_text .see ("end")


        total_processed =self .completed_count +self .failed_count 
        remaining =self .total_files -total_processed 
        progress =total_processed /self .total_files if self .total_files >0 else 0 

        self .progress_label .configure (text =f"Analyzing files... {total_processed }/{self .total_files } processed")
        self .progress_bar .set (progress )
        self .completed_label .configure (text =f"✅ Completed: {self .completed_count }")
        self .failed_label .configure (text =f"❌ Failed: {self .failed_count }")
        self .remaining_label .configure (text =f"⏳ Remaining: {remaining }")


        if total_processed >=self .total_files :
            self .finish_analysis ()

    def finish_analysis (self ):
        """Called when analysis is complete."""
        if self .cancelled :
            return 

        self .progress_label .configure (text ="✅ Analysis Complete!")
        self .cancel_button .configure (text ="Close",state ="disabled")


        self .after (1500 ,self .on_analysis_complete )

    def on_analysis_complete (self ):
        """Handle successful completion."""
        if not self .cancelled :
            self .callback_on_complete (self .analysis_results )
        self .destroy ()

    def on_cancel (self ):
        """Handle cancellation."""
        if not self .cancelled :
            self .cancelled =True 
            self .callback_on_cancel ()
            self .destroy ()

    def add_analysis_result (self ,file_path ,result ):
        """Store analysis result."""
        if result :
            self .analysis_results [file_path ]=result 




class AutoVMAFEncoderGUI (ctk .CTk ):
    def __init__ (self ):
        super ().__init__ ()

        self .title ("Auto Target Encoder - ML Enhanced Edition")
        self .geometry ("1300x850")


        self .analysis_cache ={}
        self .media_info_cache ={}
        self .model_persistence =ModelPersistence ()
        self.performance_model = PerformanceModel(self.model_persistence)
        self.sampling_time_predictor = None 
        self.search_time_predictor = None 
        self.quality_models = {}
        self.perf_error_analyzer = PerformanceErrorAnalyzer()
        self.session_encodes_completed = 0 
        self.error_analyzer = PredictionErrorAnalyzer(database_manager) if database_manager else None
        self.current_quality_model = None 

        self .config_editor =None 
        self .encoding_active =False 
        self .analysis_cancelled =False 
        self .worker_threads =[]
        self .callback_queue =queue .Queue ()
        self .video_files =[]
        self .next_file_index =0 
        self .temp_files =set ()
        self .queue_stopped =False 
        self .aborting =False 
        self .processed_files =set ()
        self .currently_processing =set ()
        self .protocol ("WM_DELETE_WINDOW",self .on_closing )
        self .after (1 ,self .initialize_app )
        self .active_analysis_processes ={}
        self .analysis_cancel_event =None 
        self .analysis_executor =None 
        self .analysis_futures =[]
        self .analysis_thread =None 

    def initialize_app (self ):

        self .load_config_to_globals ()


        self ._style_treeview ()
        self .setup_ui ()


        self .status_label .configure (text ="Settings loaded successfully",text_color ="green")


        self ._train_models_on_startup ()
        self .process_callbacks ()







    def _train_models_on_startup(self):
        """Train all ML models on startup."""
        if not database_manager:
            return

        print("--- Starting ML Model Training ---")
        try:
            # Initialize new predictor models
            self.sampling_time_predictor = SamplingTimePredictor(self.model_persistence)
            self.search_time_predictor = SearchTimePredictor(self.model_persistence)

            # Fetch performance records once for all models
            perf_records = database_manager.get_all_performance_records(limit=1000)
            
            # Train performance model for final encode (Step 3)
            print("Training final encode performance model...")
            self.performance_model.train(perf_records)
            
            # Train sampling time predictor (Step 1)
            print("Training sampling time prediction model...")
            self.sampling_time_predictor.train(perf_records)

            # Train search time predictor (Step 2)
            print("Training quality search time prediction model...")
            self.search_time_predictor.train(perf_records)

            # Train quality models (for CQ prediction)
            all_quality_records = database_manager.get_all_quality_records_for_training(limit=20000)
            if not all_quality_records:
                print("No quality records found for quality model training.")
                return

            encoders = ['nvenc', 'svt_av1']
            metrics = [('vmaf', ['average', 'percentile_1', 'percentile_5', 'percentile_10']), ('ssimulacra2', [None]), ('butteraugli', [None])]

            for encoder in encoders:
                for metric, subtypes in metrics:
                    for subtype in subtypes:
                        quality_model = QualityModel(encoder, metric, subtype, self.model_persistence)
                        model_name = quality_model._get_model_name()
                        model_id = f"{encoder.upper()}/{metric.upper()}" + (f" ({subtype})" if subtype else "")

                        if not quality_model.is_trained or self.model_persistence.get_model_age_hours(model_name) > 24:
                             print(f"Training quality model for {model_id}...")
                             quality_model.train(all_quality_records)
                        
                        self.quality_models[model_name.replace('quality_model_', '')] = quality_model
            
            print("--- ML model initialization complete ---")
        
        except Exception as e:
            import traceback
            print(f"Error training models on startup: {e}\n{traceback.format_exc()}")







    def _style_treeview (self ):
        """Style the ttk.Treeview widget."""
        style =ttk .Style ()
        style .theme_use ("clam")

        dark_bg ="#2b2b2b"
        light_text ="#DCE4EE"
        selected_blue ="#34719e"
        header_bg ="#333333"

        style .configure ("Treeview",
        background =dark_bg ,
        foreground =light_text ,
        fieldbackground =dark_bg ,
        rowheight =25 )

        style .map ('Treeview',
        background =[('selected',selected_blue )])

        style .configure ("Treeview.Heading",
        background =header_bg ,
        foreground =light_text ,
        relief ="flat")

        style .map ("Treeview.Heading",
        background =[('active','#3c3c3c')])

    def setup_ui (self ):

        self .control_frame =ctk .CTkFrame (self )
        self .control_frame .pack (fill ="x",padx =10 ,pady =10 )

        title_label =ctk .CTkLabel (
        self .control_frame ,
        text ="🎬 Auto Target Encoder - ML Enhanced",
        font =ctk .CTkFont (size =24 ,weight ="bold")
        )
        title_label .pack (side ="left",padx =10 )

        self .abort_button =ctk .CTkButton (
        self .control_frame ,
        text ="Abort Process",
        command =self .abort_process ,
        state ="disabled",
        fg_color ="red",
        hover_color ="darkred",
        font =ctk .CTkFont (weight ="bold"),
        width =150 
        )
        self .abort_button .pack (side ="right",padx =5 )

        self .stop_queue_button =ctk .CTkButton (
        self .control_frame ,
        text ="Stop Queue",
        command =self .stop_queue ,
        state ="disabled",
        fg_color ="#CC6600",
        hover_color ="#B85500",
        font =ctk .CTkFont (weight ="bold"),
        width =150 
        )
        self .stop_queue_button .pack (side ="right",padx =5 )

        self .start_button =ctk .CTkButton (
        self .control_frame ,
        text ="Start Encoding",
        command =self .start_encoding ,
        fg_color ="green",
        hover_color ="darkgreen",
        font =ctk .CTkFont (weight ="bold"),
        width =120 
        )
        self .start_button .pack (side ="right",padx =5 )

        self .settings_button =ctk .CTkButton (
        self .control_frame ,
        text ="Settings",
        command =self .open_settings ,
        font =ctk .CTkFont (weight ="bold"),
        width =100 
        )
        self .settings_button .pack (side ="right",padx =5 )


        self .tabview =ctk .CTkTabview (self )
        self .tabview .pack (fill ="both",expand =True ,padx =10 ,pady =(0 ,10 ))

        self .tab_dashboard =self .tabview .add ("📊 Dashboard")
        self .tab_queue =self .tabview .add ("📋 Queue")
        self .tab_completed =self .tabview .add ("✅ Completed")

        self .setup_dashboard_tab ()
        self .setup_queue_tab ()
        self .setup_completed_tab ()


        self .status_frame =ctk .CTkFrame (self ,height =30 )
        self .status_frame .pack (fill ="x",padx =10 ,pady =(0 ,10 ))

        self .status_label =ctk .CTkLabel (
        self .status_frame ,
        text ="Ready",
        text_color ="gray"
        )
        self .status_label .pack (side ="left",padx =10 )

        self .version_label =ctk .CTkLabel (
        self .status_frame ,
        text ="V1.0 - ML Enhanced",
        text_color ="gray"
        )
        self .version_label .pack (side ="right",padx =10 )

    def setup_dashboard_tab (self ):
        """Setup dashboard tab."""
        self .dashboard =EncodingDashboard (self .tab_dashboard ,self )
        self .dashboard .pack (fill ="both",expand =True )

    def setup_queue_tab (self ):
        """Setup queue management tab."""
        queue_frame =ctk .CTkFrame (self .tab_queue )
        queue_frame .pack (fill ="both",expand =True ,padx =10 ,pady =10 )

        ctk .CTkLabel (
        queue_frame ,
        text ="Encoding Queue",
        font =ctk .CTkFont (size =14 ,weight ="bold")
        ).pack (anchor ="w",pady =5 )


        dir_frame =ctk .CTkFrame (queue_frame )
        dir_frame .pack (fill ="x",pady =10 )

        ctk .CTkLabel (
        dir_frame ,
        text ="Input Directory",
        font =ctk .CTkFont (size =12 ,weight ="bold")
        ).pack (anchor ="w",pady =2 )

        dir_select_frame =ctk .CTkFrame (dir_frame )
        dir_select_frame .pack (fill ="x",pady =5 )

        self .input_dir_entry =ctk .CTkEntry (dir_select_frame ,width =500 )
        self .input_dir_entry .pack (side ="left",padx =5 )

        ctk .CTkButton (
        dir_select_frame ,
        text ="Browse",
        command =self .browse_input_dir ,
        width =100 
        ).pack (side ="left",padx =5 )

        ctk .CTkButton (
        dir_select_frame ,
        text ="Scan",
        command =self .scan_videos ,
        width =100 
        ).pack (side ="left",padx =5 )


        controls_frame =ctk .CTkFrame (queue_frame )
        controls_frame .pack (fill ="x",pady =10 )

        ctk .CTkButton (
        controls_frame ,
        text ="Clear Queue",
        command =self .clear_queue ,
        width =120 
        ).pack (side ="left",padx =5 )

        ctk .CTkButton (
        controls_frame ,
        text ="Add Files",
        command =self .add_files_to_queue ,
        width =120 
        ).pack (side ="left",padx =5 )


        self .queue_tree_frame =ctk .CTkFrame (queue_frame )
        self .queue_tree_frame .pack (fill ="both",expand =True ,pady =5 )

        queue_scroll_y =ctk .CTkScrollbar (self .queue_tree_frame )
        queue_scroll_y .pack (side ="right",fill ="y")

        self .queue_tree =ttk .Treeview (
        self .queue_tree_frame ,
        yscrollcommand =queue_scroll_y .set ,
        columns =("size","status"),
        show ="tree headings",
        height =15 
        )
        self .queue_tree .pack (fill ="both",expand =True )

        queue_scroll_y .configure (command =self .queue_tree .yview )

        self .queue_tree .heading ("#0",text ="File Name")
        self .queue_tree .heading ("size",text ="Size")
        self .queue_tree .heading ("status",text ="Status")

        self .queue_tree .column ("#0",width =500 )
        self .queue_tree .column ("size",width =100 )
        self .queue_tree .column ("status",width =100 )

        self .queue_stats_label =ctk .CTkLabel (
        queue_frame ,
        text ="Queue: 0 files | 0 MB",
        font =ctk .CTkFont (size =12 )
        )
        self .queue_stats_label .pack (anchor ="w",pady =5 )


    def setup_completed_tab (self ):
        """Setup completed files tab."""
        completed_frame =ctk .CTkFrame (self .tab_completed )
        completed_frame .pack (fill ="both",expand =True ,padx =10 ,pady =10 )

        header_frame =ctk .CTkFrame (completed_frame )
        header_frame .pack (fill ="x",pady =(0 ,10 ))

        ctk .CTkLabel (
        header_frame ,
        text ="Completed Files",
        font =ctk .CTkFont (size =14 ,weight ="bold")
        ).pack (side ="left",padx =10 ,pady =5 )

        controls_frame =ctk .CTkFrame (header_frame )
        controls_frame .pack (side ="right",padx =10 ,pady =5 )

        self .filter_var =ctk .StringVar (value ="All")
        filter_menu =ctk .CTkComboBox (
        controls_frame ,
        variable =self .filter_var ,
        values =["All","Success","Skipped","Failed"],
        command =self .filter_completed_files ,
        width =120
        )
        filter_menu .pack (side ="left",padx =5 )

        ctk .CTkButton (
        controls_frame ,
        text ="Export CSV",
        command =self .export_completed_to_csv ,
        width =100 ,
        font =ctk .CTkFont (weight ="bold")
        ).pack (side ="left",padx =5 )

        ctk .CTkButton (
        controls_frame ,
        text ="Clear List",
        command =self .clear_completed_list ,
        width =100
        ).pack (side ="left",padx =5 )


        self .completed_tree_frame =ctk .CTkFrame (completed_frame )
        self .completed_tree_frame .pack (fill ="both",expand =True ,pady =5 )

        completed_scroll_y =ctk .CTkScrollbar (self .completed_tree_frame )
        completed_scroll_y .pack (side ="right",fill ="y")

        completed_scroll_x =ctk .CTkScrollbar (self .completed_tree_frame ,orientation ="horizontal")
        completed_scroll_x .pack (side ="bottom",fill ="x")

        self .completed_tree =ttk .Treeview (
        self .completed_tree_frame ,
        yscrollcommand =completed_scroll_y .set ,
        xscrollcommand =completed_scroll_x .set ,
        columns =(
        "status","initial_size","output_size","size_reduction","compression_ratio",
        "final_cq","processing_duration","final_quality","ml_accelerated","skip_reason"
        ),
        show ="tree headings",
        height =20
        )
        self .completed_tree .pack (fill ="both",expand =True )

        completed_scroll_y .configure (command =self .completed_tree .yview )
        completed_scroll_x .configure (command =self .completed_tree .xview )


        columns_config ={
        "#0":("File Name",300 ),
        "status":("Status",80 ),
        "initial_size":("Initial Size",90 ),
        "output_size":("Output Size",90 ),
        "size_reduction":("Size Reduction %",110 ),
        "compression_ratio":("Compression",90 ),
        "final_cq":("Final CQ",70 ),
        "processing_duration":("Duration",80 ),
        "final_quality":("Quality Score",100 ),
        "ml_accelerated":("ML Accelerated",100 ),
        "skip_reason":("Details / Reason for Skip", 300)
        }

        for col_id ,(header ,width )in columns_config .items ():
            self .completed_tree .heading (col_id ,text =header )
            self .completed_tree .column (col_id ,width =width ,minwidth =50 )

        self .summary_label =ctk .CTkLabel (
        completed_frame ,
        text ="Summary: 0 files completed | 0 MB saved | 0% average reduction",
        font =ctk .CTkFont (size =12 ,weight ="bold")
        )
        self .summary_label .pack (padx =10 ,pady =5 )

        self .completed_files =[]



    def open_settings (self ):
        """Open settings editor."""
        if self .config_editor is None or not self .config_editor .winfo_exists ():
            self .config_editor =ConfigEditor (self )
            self .config_editor .grab_set ()

    def refresh_dashboard (self ):
        """Refresh dashboard after settings change."""
        self .load_config_to_globals ()
        if hasattr (self ,'dashboard'):
            self .dashboard .destroy ()
            self .dashboard =EncodingDashboard (self .tab_dashboard ,self )
            self .dashboard .pack (fill ="both",expand =True )

    def browse_input_dir (self ):
        """Browse for input directory."""
        directory =filedialog .askdirectory ()
        if directory :
            self .input_dir_entry .delete (0 ,tk .END )
            self .input_dir_entry .insert (0 ,directory )

    def scan_videos (self ):
        """Scan for video files."""
        input_dir =self .input_dir_entry .get ()
        if not input_dir :
            messagebox .showwarning ("Warning","Please select an input directory first.")
            return 

        if not SETTINGS :
            self .load_config_to_globals ()

        self .video_files =[]
        for item in self .queue_tree .get_children ():
            self .queue_tree .delete (item )

        video_extensions ={
        '.mp4','.mkv','.mov','.webm','.avi','.flv',
        '.wmv','.ts','.m2ts','.mpg','.mpeg','.m4v','.y4m'
        }

        total_size =0 

        for file_path in Path (input_dir ).rglob ('*'):
            if file_path .suffix .lower ()in video_extensions :

                if (SETTINGS .output_suffix in file_path .stem or 
                SETTINGS .skipped_file_suffix in file_path .stem ):
                    continue 

                self .video_files .append (str (file_path ))

                try :
                    size_mb =file_path .stat ().st_size /(1024 *1024 )
                    total_size +=size_mb 

                    self .queue_tree .insert (
                    "",
                    "end",
                    text =file_path .name ,
                    values =(f"{size_mb :.1f} MB","Pending")
                    )
                except :
                    self .queue_tree .insert (
                    "",
                    "end",
                    text =file_path .name ,
                    values =("Unknown","Pending")
                    )

        self .queue_stats_label .configure (
        text =f"Queue: {len (self .video_files )} files | {total_size :.1f} MB"
        )

        self .dashboard .log_message (f"Found {len (self .video_files )} video files","info")
        self .status_label .configure (text =f"Found {len (self .video_files )} video files")

    def clear_queue (self ):
        """Clear the encoding queue."""
        self .video_files =[]
        for item in self .queue_tree .get_children ():
            self .queue_tree .delete (item )
        self .queue_stats_label .configure (text ="Queue: 0 files | 0 MB")
        self .dashboard .log_message ("Queue cleared","info")

    def add_files_to_queue (self ):
        """Add individual files to queue."""
        files =filedialog .askopenfilenames (
        filetypes =[
        ("Video files","*.mp4 *.mkv *.mov *.webm *.avi"),
        ("All files","*.*")
        ]
        )
        if files :
            for file in files :
                self .video_files .append (file )
                try :
                    size_mb =os .path .getsize (file )/(1024 *1024 )
                    self .queue_tree .insert (
                    "",
                    "end",
                    text =os .path .basename (file ),
                    values =(f"{size_mb :.1f} MB","Pending")
                    )
                except :
                    self .queue_tree .insert (
                    "",
                    "end",
                    text =os .path .basename (file ),
                    values =("Unknown","Pending")
                    )

            self .update_queue_stats ()
            self .dashboard .log_message (f"Added {len (files )} files to queue","info")



    def log_prefiltered_skip(self, file_path, reason):
        """Logs a file skipped during the pre-filter phase."""
        # 1. Add to GUI's completed tab
        self.add_file_to_completed(file_path, 'Skipped', skip_reason=f"Pre-filtered: {reason}")

        # 2. Add a simple entry to encoding_log.txt
        try:
            if SETTINGS and SETTINGS.encoding_log_path:
                log_dir = os.path.dirname(SETTINGS.encoding_log_path)
                if log_dir and not os.path.exists(log_dir):
                    os.makedirs(log_dir, exist_ok=True)

                timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
                log_entry = (
                    f"--- Encoding Log ({timestamp}) ---\n"
                    f"File: {Path(file_path).name}\n"
                    f"Status: Skipped\n"
                    f"Reason: Pre-filtered - {reason}\n"
                    f"----------------------------------------\n\n"
                )

                with open(SETTINGS.encoding_log_path, 'a', encoding='utf-8') as f:
                    f.write(log_entry)
        except Exception as e:
            print(f"Failed to write pre-filter skip to log file: {e}")



    def update_queue_stats (self ):
        """Update queue statistics."""
        total_size =0 
        for file_path in self .video_files :
            try :
                total_size +=os .path .getsize (file_path )/(1024 *1024 )
            except :
                pass 

        self .queue_stats_label .configure (
        text =f"Queue: {len (self .video_files )} files | {total_size :.1f} MB"
        )



    def update_queue_item_status (self ,file_path :str ,status :str ):
        """Mettre à jour le statut d'un élément dans la file d'attente."""
        filename =os .path .basename (file_path )


        for item in self .queue_tree .get_children ():
            item_text =self .queue_tree .item (item ,"text")
            if item_text ==filename :
                if status in ["Completed","Failed","Skipped"]:

                    self .queue_tree .delete (item )

                    if not hasattr (self ,'processed_files'):
                        self .processed_files =set ()
                    self .processed_files .add (file_path )
                else :

                    current_values =list (self .queue_tree .item (item ,"values"))
                    current_values [1 ]=status 
                    self .queue_tree .item (item ,values =current_values )
                break 


        self .update_remaining_queue_stats ()


    def update_remaining_queue_stats (self ):
        """Update queue statistics showing only remaining files."""
        if not hasattr (self ,'processed_files'):
            self .processed_files =set ()

        remaining_files =[f for f in self .video_files if f not in self .processed_files ]
        total_size =0 

        for file_path in remaining_files :
            try :
                total_size +=os .path .getsize (file_path )/(1024 *1024 )
            except :
                pass 

        self .queue_stats_label .configure (
        text =f"Queue: {len (remaining_files )} files remaining | {total_size :.1f} MB"
        )




    def get_next_unprocessed_file (self ):
        """Get the next file that hasn't been processed yet."""
        if not hasattr (self ,'processed_files'):
            self .processed_files =set ()
        if not hasattr (self ,'currently_processing'):
            self .currently_processing =set ()

        for file_path in self .video_files :
            if (file_path not in self .processed_files and 
            file_path not in self .currently_processing and 
            file_path in self .analysis_cache ):

                self .currently_processing .add (file_path )
                return file_path 

        return None 




    def pre_filter_files_with_progress(self, callback_on_complete):
        """Runs a fast pre-filter on all files silently in the background."""
        files_to_check = self.video_files[:]
        if not files_to_check:
            self.after(0, callback_on_complete)
            return

        valid_files = []
        skipped_files = []

        def pre_filter_thread():
            # This thread now runs completely silently without a GUI.
            size_filter_enabled = SETTINGS.min_filesize_mb > 0
            duration_filter_enabled = SETTINGS.min_duration_seconds > 0
            bitrate_filters_enabled = any([
                SETTINGS.min_bitrate_4k_kbps > 0,
                SETTINGS.min_bitrate_1080p_kbps > 0,
                SETTINGS.min_bitrate_720p_kbps > 0
            ])
            ffprobe_needed = duration_filter_enabled or bitrate_filters_enabled

            for i, file_path in enumerate(files_to_check):
                if self.analysis_cancelled:
                    break

                reason = ""
                try:
                    if size_filter_enabled:
                        size_mb = os.path.getsize(file_path) / (1024 * 1024)
                        if size_mb < SETTINGS.min_filesize_mb:
                            reason = f"File size too small ({size_mb:.1f}MB < {SETTINGS.min_filesize_mb}MB)"

                    if not reason and ffprobe_needed:
                        media_info = get_media_info(file_path)
                        if not media_info:
                            reason = "Could not read media info"
                        else:
                            duration = float(media_info.get('format', {}).get('duration', 0))
                            if duration_filter_enabled and duration < SETTINGS.min_duration_seconds:
                                reason = f"Duration too short ({duration:.1f}s < {SETTINGS.min_duration_seconds}s)"
                            else:
                                video_stream = next((s for s in media_info.get('streams', []) if s.get('codec_type') == 'video'), None)
                                if bitrate_filters_enabled and video_stream:
                                    width = int(video_stream.get('width', 0))
                                    bitrate_kbps = float(media_info.get('format', {}).get('bit_rate', '0')) / 1000
                                    if width >= 3840 and SETTINGS.min_bitrate_4k_kbps > 0 and bitrate_kbps < SETTINGS.min_bitrate_4k_kbps:
                                        reason = f"4K bitrate too low ({bitrate_kbps:.0f} < {SETTINGS.min_bitrate_4k_kbps} kbps)"
                                    elif 1920 <= width < 3840 and SETTINGS.min_bitrate_1080p_kbps > 0 and bitrate_kbps < SETTINGS.min_bitrate_1080p_kbps:
                                        reason = f"1080p bitrate too low ({bitrate_kbps:.0f} < {SETTINGS.min_bitrate_1080p_kbps} kbps)"
                                    elif 1280 <= width < 1920 and SETTINGS.min_bitrate_720p_kbps > 0 and bitrate_kbps < SETTINGS.min_bitrate_720p_kbps:
                                        reason = f"720p bitrate too low ({bitrate_kbps:.0f} < {SETTINGS.min_bitrate_720p_kbps} kbps)"

                    if reason:
                        skipped_files.append(file_path)
                        self.after(0, lambda p=file_path, r=reason: self.log_prefiltered_skip(p, r))
                        rename_skipped_file(file_path, 0, lambda id, msg: print(msg), reason_suffix=SETTINGS.skipped_file_filter_suffix)
                    else:
                        valid_files.append(file_path)
                except Exception as e:
                    reason = f"Error: {e}"
                    skipped_files.append(file_path)

            # When done, update the main file list and trigger the next step.
            if not self.analysis_cancelled:
                self.video_files = valid_files
                self.after(0, self.update_queue_list_after_filter)
                self.after(100, callback_on_complete)

        threading.Thread(target=pre_filter_thread, daemon=True).start()

    # ADD THIS NEW HELPER METHOD
    def update_queue_list_after_filter(self):
        """Refreshes the queue Treeview after pre-filtering is complete."""
        for item in self.queue_tree.get_children():
            self.queue_tree.delete(item)

        total_size = 0
        for file_path in self.video_files:
            try:
                size_mb = os.path.getsize(file_path) / (1024 * 1024)
                total_size += size_mb
                self.queue_tree.insert("", "end", text=os.path.basename(file_path), values=(f"{size_mb:.1f} MB", "Pending"))
            except OSError:
                self.queue_tree.insert("", "end", text=os.path.basename(file_path), values=("N/A", "Pending"))

        self.queue_stats_label.configure(text=f"Queue: {len(self.video_files)} valid files | {total_size:.1f} MB")




    def load_config_to_globals (self ):
        """Load configuration with robust fallback handling."""
        global SETTINGS ,database_manager ,memory_manager ,FFMPEG_ENV 

        config =configparser .ConfigParser ()


        if not os .path .exists ('config.ini'):
            if not self .create_minimal_config ():
                return False 

        config .read ('config.ini',encoding ='utf-8')


        def safe_get (section ,key ,fallback ='',required =False ):
            """Safely get string value from config."""
            try :
                if config .has_section (section )and config .has_option (section ,key ):
                    value =config .get (section ,key ).strip ()
                    if value :
                        return value 
            except :
                pass 

            if required and not fallback :
                raise ValueError (f"Required setting missing: [{section }] {key }")
            return fallback 

        def safe_getint (section ,key ,fallback =0 ,min_val =None ,max_val =None ):
            """Safely get integer value from config."""
            try :
                if config .has_section (section )and config .has_option (section ,key ):
                    value_str =config .get (section ,key ).strip ()
                    if value_str :
                        value =int (value_str )
                        if min_val is not None :
                            value =max (min_val ,value )
                        if max_val is not None :
                            value =min (max_val ,value )
                        return value 
            except (ValueError ,TypeError ):
                pass 
            return fallback 

        def safe_getfloat (section ,key ,fallback =0.0 ,min_val =None ,max_val =None ):
            """Safely get float value from config."""
            try :
                if config .has_section (section )and config .has_option (section ,key ):
                    value_str =config .get (section ,key ).strip ()
                    if value_str :
                        value =float (value_str )
                        if min_val is not None :
                            value =max (min_val ,value )
                        if max_val is not None :
                            value =min (max_val ,value )
                        return value 
            except (ValueError ,TypeError ):
                pass 
            return fallback 

        def safe_getboolean (section ,key ,fallback =False ):
            """Safely get boolean value from config."""
            try :
                if config .has_section (section )and config .has_option (section ,key ):
                    value_str =config .get (section ,key ).strip ().lower ()
                    if value_str :

                        if value_str in ('true','1','yes','on'):
                            return True 
                        elif value_str in ('false','0','no','off',''):
                            return False 
            except :
                pass 
            return fallback 

        try :

            encoder_type =safe_get ('Encoder','encoder_type','nvenc').lower ()


            SETTINGS =EncodingSettings (

            ffmpeg_path =safe_get ('Paths','ffmpeg_path',required =True ),
            ffprobe_path =safe_get ('Paths','ffprobe_path',required =True ),


            database_path =safe_get ('Paths','database_path','encoding_cache.db'),
            encoding_log_path =safe_get ('Paths','encoding_log_path','encoding_log.txt'),
            ffvship_path =safe_get ('Paths','ffvship_path','ffvship.exe'),


            max_workers =safe_getint ('Performance','max_workers',1 ,min_val =1 ,max_val =16 ),
            num_parallel_vmaf_runs =safe_getint ('Performance','num_parallel_vmaf_runs',3 ,min_val =1 ,max_val =10 ),
            max_iterations =safe_getint ('Performance','max_iterations',7 ,min_val =3 ,max_val =20 ),
            cpu_threads =safe_getint ('Performance','cpu_threads',0 ,min_val =0 ),


            encoder_type =encoder_type ,


            nvenc_preset =safe_get ('NVENC','nvenc_preset','p4')if encoder_type =='nvenc'else 'p4',
            nvenc_quality_mode =safe_get ('NVENC','nvenc_quality_mode','quality')if encoder_type =='nvenc'else 'quality',
            nvenc_advanced_params =safe_get ('NVENC_Advanced','extra_params','')if encoder_type =='nvenc'else '',


            svt_av1_preset =safe_getint ('SVT_AV1','svt_av1_preset',7 ,min_val =0 ,max_val =13 )if encoder_type =='svt_av1'else 7 ,
            svt_av1_advanced_params =safe_get ('SVT_AV1_Advanced','extra_params','')if encoder_type =='svt_av1'else '',


            quality_metric_mode =safe_get ('Quality_Metrics','quality_metric_mode','vmaf').lower (),
            target_score =safe_getfloat ('Quality_Metrics','target_score',94.0 ,min_val =0 ,max_val =100 ),
            quality_tolerance_percent =safe_getfloat ('Quality_Metrics','quality_tolerance_percent',0.5 ,min_val =0.1 ,max_val =10 ),
            cq_search_min =safe_getint ('Quality_Metrics','cq_search_min',20 ,min_val =0 ,max_val =63 ),
            cq_search_max =safe_getint ('Quality_Metrics','cq_search_max',35 ,min_val =0 ,max_val =63 ),
            vmaf_targeting_mode =safe_get ('Quality_Metrics','vmaf_targeting_mode','average').lower (),
            vmaf_target_percentile =safe_getfloat ('Quality_Metrics','vmaf_target_percentile',1.0 ,min_val =0.1 ,max_val =99 ),


            sampling_method =safe_get ('Quality_Sampling','sampling_method','tier1').lower (),
            sample_segment_duration =safe_getint ('Quality_Sampling','sample_segment_duration',3 ,min_val =1 ,max_val =30 ),
            num_samples =safe_getint ('Quality_Sampling','num_samples',4 ,min_val =1 ,max_val =20 ),
            master_sample_encoder =safe_get ('Quality_Sampling','master_sample_encoder','software').lower (),
            min_scene_changes_required =safe_getint ('Quality_Sampling','min_scene_changes_required',5 ,min_val =1 ),
            min_keyframes_required =safe_getint ('Quality_Sampling','min_keyframes_required',5 ,min_val =1 ),
            skip_start_seconds =safe_getint ('Quality_Sampling','skip_start_seconds',0 ,min_val =0 ),
            skip_end_seconds =safe_getint ('Quality_Sampling','skip_end_seconds',0 ,min_val =0 ),
            ffmpeg_scenedetect_threshold =safe_getfloat ('Quality_Sampling','ffmpeg_scenedetect_threshold',0.4 ,min_val =0.0 ,max_val =1.0 ),


            min_duration_seconds =safe_getint ('File_Filtering','min_duration_seconds',0 ,min_val =0 ),
            min_filesize_mb =safe_getint ('File_Filtering','min_filesize_mb',0 ,min_val =0 ),
            min_bitrate_4k_kbps =safe_getint ('File_Filtering','min_bitrate_4k_kbps',0 ,min_val =0 ),
            min_bitrate_1080p_kbps =safe_getint ('File_Filtering','min_bitrate_1080p_kbps',0 ,min_val =0 ),
            min_bitrate_720p_kbps =safe_getint ('File_Filtering','min_bitrate_720p_kbps',0 ,min_val =0 ),


            enable_quality_cache =safe_getboolean ('Cache','enable_quality_cache',True ),
            enable_performance_log =safe_getboolean ('Cache','enable_performance_log',True ),


            delete_source_file =safe_getboolean ('File_Management','delete_source_file',False ),
            output_suffix =safe_get ('File_Management','output_suffix','_av1'),
            output_directory =safe_get ('File_Management','output_directory',''),
            use_different_input_directory =safe_getboolean ('File_Management','use_different_input_directory',False ),
            input_directory =safe_get ('File_Management','input_directory',''),
            min_size_reduction_threshold =safe_getfloat ('File_Management','min_size_reduction_threshold',5.0 ,min_val =0 ,max_val =100 ),
            rename_skipped_files =safe_getboolean ('File_Management','rename_skipped_files',False ),
            skipped_file_suffix =safe_get ('File_Management','skipped_file_suffix','_notencoded'),
            skipped_file_filter_suffix=safe_get('File_Management', 'skipped_file_filter_suffix', '_Skipped_FileFilterSettings'),
            skip_encoding_if_target_not_reached =safe_getboolean ('File_Management','skip_encoding_if_target_not_reached',False ),


            output_bit_depth =safe_get ('Output','output_bit_depth','source').lower (),
            ml_extra_cq_check =safe_getboolean ('Quality_Metrics','ml_extra_cq_check',False )
            )


            if not os .path .exists (SETTINGS .ffmpeg_path ):
                messagebox .showerror ("Error",f"FFmpeg not found at: {SETTINGS .ffmpeg_path }")
                return False 

            if not os .path .exists (SETTINGS .ffprobe_path ):
                messagebox .showerror ("Error",f"FFprobe not found at: {SETTINGS .ffprobe_path }")
                return False 


            try :
                if SETTINGS .enable_quality_cache or SETTINGS .enable_performance_log :
                    database_manager =DatabaseManager (SETTINGS .database_path )
                else :
                    database_manager =None 
            except Exception as e :
                print (f"Warning: Database initialization failed: {e }")
                database_manager =None 

            memory_manager =MemoryManager ()
            FFMPEG_ENV =get_ffmpeg_env ()


            return True 

        except ValueError as e :
            messagebox .showerror ("Configuration Error",str (e ))
            return False 
        except Exception as e :
            messagebox .showerror ("Error",f"Failed to load settings: {e }")
            return False 




    def create_minimal_config (self ):
        """Create a minimal config.ini with only required settings."""
        minimal_config ="""[Paths]
ffmpeg_path = 
ffprobe_path = 

[Encoder]
encoder_type = nvenc

[Quality_Metrics]
quality_metric_mode = vmaf
target_score = 94
cq_search_min = 20
cq_search_max = 35
"""

        try :
            with open ('config.ini','w')as f :
                f .write (minimal_config )

            messagebox .showinfo (
            "Config Created",
            "A minimal config.ini has been created.\n"
            "Please set the FFmpeg and FFprobe paths in Settings."
            )
            return True 
        except Exception as e :
            messagebox .showerror ("Error",f"Failed to create config.ini: {e }")
            return False 




    def start_encoding (self ):
        """Start the encoding process with ML acceleration."""
        if not self .video_files :
            messagebox .showwarning ("Warning","No video files in queue. Please scan for videos first.")
            return

        self .load_config_to_globals ()
        if SETTINGS is None :
            messagebox .showerror ("Error","Settings not loaded. Please check configuration.")
            return

        self .encoding_active =True
        self .queue_stopped =False
        self.analysis_cancelled = False
        self .analysis_cache .clear ()
        self .start_button .configure (state ="disabled")
        self .stop_queue_button .configure (state ="disabled")
        self .abort_button .configure (state ="normal")
        self .settings_button .configure (state ="disabled")
        self .tabview .set ("📊 Dashboard")
        self .dashboard .reset ()
        self .dashboard .log_message ("Starting encoding process...","info")

        if database_manager :
            self .dashboard .log_message ("Checking for model updates...","info")
            threading .Thread (target =self ._train_models_on_startup ,daemon =True ).start ()
        
        # MODIFIED: Update status bar instead of creating a dialog
        self.status_label.configure(text="Running fast pre-filter in background...", text_color="orange")
        self.dashboard.log_message(f"Pre-filtering {len(self.video_files)} files...", "info")
        
        # MODIFIED: We now pass the callback to the next step.
        self.pre_filter_files_with_progress(self._start_main_analysis_after_filter)



    def _start_main_analysis_after_filter(self):
        """This is called after pre-filtering is complete to start the main analysis."""
        if not self.video_files:
            messagebox.showinfo("Information", "No valid files remaining after filtering.")
            self._reset_ui_after_stop()
            self.encoding_active = False
            return

        self.dashboard.log_message(f"Starting main analysis for {len(self.video_files)} valid files...", "info")
        self.run_analysis_with_progress(self._on_analysis_complete, self._on_analysis_cancelled)





    def _on_analysis_complete (self ,analysis_results ):

        valid_results = {}
        failed_files = {}

        for file_path, result in analysis_results.items():
            if result.get('status') == 'analysis_failed':
                failed_files[file_path] = result.get('reason', 'Unknown analysis error')
            else:
                valid_results[file_path] = result

        if failed_files:
            self.dashboard.log_message(f"{len(failed_files)} file(s) failed analysis and will be skipped.", "warning")
            for file_path, reason in failed_files.items():
                self.add_file_to_completed(file_path, 'Skipped', skip_reason=f"Analysis failed: {reason}")
                self.update_queue_item_status(file_path, "Skipped")
                rename_skipped_file(file_path, 0, lambda id, msg: print(f"[Rename]: {msg}"), reason_suffix=SETTINGS.skipped_file_suffix)

        """Callback executed after file analysis is successful."""
        self .analysis_cache = valid_results

        if not self .analysis_cache :
            messagebox .showerror ("Error","No files could be analyzed. Check your video files.")
            self ._on_analysis_cancelled ()
            return 

        self .dashboard .log_message ("Analysis complete. Starting encoding...","info")
        self .dashboard .start_time =time .time ()


        self .stop_queue_button .configure (state ="normal")
        self .abort_button .configure (state ="normal")



        self .batch_state ={
        'files_completed':0 ,

        'total_files':len ([f for f in self .video_files if f not in self .processed_files ]),
        'start_time':time .time (),
        'deleted_source_size':0 ,
        'encoded_output_size':0 
        }



        self .currently_processing .clear ()
        self .worker_threads .clear ()


        for _ in range (SETTINGS .max_workers ):
            self ._launch_next_worker ()


        self .calculate_ml_based_eta ()




    def _on_analysis_cancelled (self ):
        """Handle cancellation by killing all active analysis subprocesses."""

        if hasattr (self ,'analysis_cancel_event')and not self .analysis_cancel_event .is_set ():
            self .analysis_cancel_event .set ()


        self ._kill_ffmpeg_processes ()


        self .dashboard .log_message ("Analysis aborted by user","error")
        self .encoding_active =False 
        self .queue_stopped =False 

        self .start_button .configure (state ="normal")
        self .stop_queue_button .configure (state ="disabled")
        self .abort_button .configure (state ="disabled")
        self .settings_button .configure (state ="normal")

        self .dashboard .reset ()
        self .status_label .configure (text ="Analysis aborted")




    def calculate_ml_based_eta(self):
        """Calculate ETA using dedicated models for each step of the process."""
        if not hasattr(self, 'video_files') or not self.video_files:
            return

        remaining_files = [f for f in self.video_files if f not in self.processed_files and f not in self.currently_processing]
        if not remaining_files and not self.currently_processing:
            self.dashboard.eta_label.configure(text="ETA: Complete!", text_color="green")
            return

        if not self.sampling_time_predictor or not self.search_time_predictor:
            print("ETA calculation skipped: Predictor models not ready.")
            return

        feature_extractor = FeatureExtractor()
        file_etas = []

        for file_path in remaining_files:
            try:
                cached_data = self.analysis_cache.get(file_path)
                if not cached_data or 'media_info' not in cached_data:
                    file_etas.append(300)  # Default 5 mins for unanalyzed files
                    continue

                # --- Extract Features ---
                media_info = cached_data['media_info']
                complexity_data = cached_data.get('complexity_data', {})
                file_features = feature_extractor.extract_video_features(media_info, complexity_data)
                encoder_features = feature_extractor.get_encoder_features(SETTINGS)
                
                duration = file_features.get('duration_seconds', 0)
                if duration <= 0:
                    file_etas.append(300)
                    continue
                
                # --- Predict Time for Each Step ---
                total_sample_duration = SETTINGS.num_samples * SETTINGS.sample_segment_duration

                # Step 1: Predict Sampling Time
                sampling_features = {
                    'total_sample_duration_s': total_sample_duration,
                    'resolution_pixels': file_features.get('resolution_pixels', 0),
                    'source_bitrate_kbps': file_features.get('source_bitrate_kbps', 0),
                    'is_nvenc_sample_encoder': 1 if SETTINGS.master_sample_encoder == 'nvenc' else 0
                }
                predicted_sample_time = self.sampling_time_predictor.predict(sampling_features)

                # Step 2: Predict CQ Search Time (assuming 2 iterations for speed)
                search_features = {
                    'search_iterations': 2, # Optimistic assumption due to ML quality model
                    'resolution_pixels': file_features.get('resolution_pixels', 0),
                    'total_sample_duration_s': total_sample_duration
                }
                predicted_search_time = self.search_time_predictor.predict(search_features)

                # Step 3: Predict Final Encode Time
                predicted_fps, confidence = self.performance_model.predict_fps(file_features, encoder_features)
                corrected_fps = self.perf_error_analyzer.get_corrected_fps(predicted_fps, file_features, encoder_features)
                
                if corrected_fps > 0:
                    total_frames = duration * file_features.get('frame_rate', 30.0)
                    predicted_encode_time = total_frames / corrected_fps
                else:
                    predicted_encode_time = duration * 1.5 # Fallback

                # --- Combine and Finalize ---
                total_file_eta = predicted_sample_time + predicted_search_time + predicted_encode_time
                file_etas.append(total_file_eta)

            except Exception as e:
                print(f"Error calculating ETA for file {os.path.basename(file_path)}: {e}")
                file_etas.append(300)

        # --- Calculate Total Queue ETA (handling parallel workers) ---
        total_remaining_time = 0
        if SETTINGS.max_workers > 1 and len(file_etas) > 0:
            file_etas.sort(reverse=True)
            worker_times = [0] * SETTINGS.max_workers
            for eta in file_etas:
                min_worker_idx = worker_times.index(min(worker_times))
                worker_times[min_worker_idx] += eta
            total_remaining_time = max(worker_times)
        else:
            total_remaining_time = sum(file_etas)

        if total_remaining_time > 0:
            self.dashboard.initial_eta_seconds = total_remaining_time
            self.dashboard.countdown_start_time = time.time()
            self.dashboard.stop_countdown()
            self.dashboard._update_countdown()
            self.dashboard.log_message(f"Data-driven ETA calculated for all steps.", "info")
        else:
            self.dashboard.eta_label.configure(text="ETA: Calculating...", text_color="gray")






    def stop_queue (self ):
        """Stop processing queue."""
        if messagebox .askyesno ("Confirm","Stop processing queue? Current files will finish."):
            self .queue_stopped =True 
            self .encoding_active =False 


            self .stop_queue_button .configure (state ="disabled")
            self .start_button .configure (state ="disabled")

            self .dashboard .log_message ("Queue processing stopped","warning")
            self .status_label .configure (text ="Stopping queue...")


            if len (self .worker_threads )==0 :
                self .after (100 ,self ._finalize_encoding_process )

    def abort_process (self ):
        """Abort all processes."""
        if self .encoding_active and messagebox .askyesno ("Confirm","Encoding in progress. Abort all processes?"):
            self ._on_encoding_aborted ()
        elif hasattr (self ,'analysis_dialog')and self .analysis_dialog .winfo_exists ():
            if messagebox .askyesno ("Confirm","Analysis in progress. Abort and exit?"):
                self .analysis_dialog .on_cancel ()
        elif not self .encoding_active :

            self ._reset_ui_after_stop ()

    def _on_encoding_aborted (self ):
        """Logic for aborting a running encoding process."""
        self .status_label .configure (text ="Aborting processes...")
        self .dashboard .log_message ("Aborting all processes...","error")

        self .aborting =True 
        self .encoding_active =False 
        self .queue_stopped =False 


        self ._kill_ffmpeg_processes ()


        for worker in self .worker_threads :
            worker .stop ()
            worker .aborted =True 


        time .sleep (0.2 )
        self ._kill_ffmpeg_processes ()


        while not self .callback_queue .empty ():
            try :
                self .callback_queue .get_nowait ()
            except queue .Empty :
                break 


        for worker_id in self .dashboard .worker_panels :
            self .dashboard .deactivate_worker (worker_id ,"❌ Process Aborted")
            self .dashboard .log_worker_message (worker_id ,"Process aborted by user")


        self ._cleanup_temp_files ()


        self .worker_threads .clear ()
        self .dashboard .stop_countdown ()
        self .clear_media_cache ()


        self ._reset_ui_after_stop ()


        self .dashboard .log_message ("All processes aborted","error")
        self .status_label .configure (text ="Process aborted")

        self .aborting =False 

    def _kill_ffmpeg_processes (self ):
        """Kill all FFmpeg processes."""
        try :
            import psutil 
            for proc in psutil .process_iter (['pid','name','ppid']):
                try :
                    if proc .info ['name']and ('ffmpeg'in proc .info ['name'].lower ()or 
                    'ffvship'in proc .info ['name'].lower ()):

                        if proc .info ['ppid']==os .getpid ()or proc .info ['pid']in [p .pid for p in self .active_analysis_processes .values ()]:
                            proc .kill ()
                            self .dashboard .log_message (f"Killed {proc .info ['name']} process","warning")
                except (psutil .NoSuchProcess ,psutil .AccessDenied ,psutil .ZombieProcess ):
                    pass 
        except Exception as e :
            self .dashboard .log_message (f"Error killing processes: {e }","warning")

    def _cleanup_temp_files (self ):
        """Clean up temporary files."""
        cleaned_count =0 

        for temp_file in list (self .temp_files ):
            try :
                if os .path .exists (temp_file ):
                    os .unlink (temp_file )
                    cleaned_count +=1 
                self .temp_files .discard (temp_file )
            except Exception as e :
                self .dashboard .log_message (f"Could not delete {temp_file }: {e }","warning")


        try :
            if SETTINGS and SETTINGS .output_directory :
                output_dir =Path (SETTINGS .output_directory )
                if output_dir .exists ():
                    for temp_file in output_dir .glob ("*_temp*"):
                        try :
                            temp_file .unlink ()
                            cleaned_count +=1 
                        except :
                            pass 
        except :
            pass 

        if cleaned_count >0 :
            self .dashboard .log_message (f"Cleaned up {cleaned_count } temporary files","info")

    def _reset_ui_after_stop (self ):
        """Reset UI after stopping."""
        self .start_button .configure (state ="normal")
        self .stop_queue_button .configure (state ="disabled")
        self .abort_button .configure (state ="disabled")
        self .settings_button .configure (state ="normal")


        for worker_id in self .dashboard .worker_panels :
            self .dashboard .deactivate_worker (worker_id ,"Idle")


        self .dashboard .overall_progress .set (0 )

    def _finalize_encoding_process (self ):
        """Finalize encoding process."""

        was_queue_stopped =self .queue_stopped 

        self .encoding_active =False 
        self .queue_stopped =False 
        self .dashboard .stop_countdown ()
        self .dashboard .eta_label .configure (text ="ETA: Complete!",text_color ="green")


        self .stop_queue_button .configure (state ="disabled")
        self .abort_button .configure (state ="disabled")

        self ._reset_ui_after_stop ()


        if self .dashboard .start_time is not None :
            elapsed =time .time ()-self .dashboard .start_time 
            if was_queue_stopped :
                message ="Queue processing stopped"
                status_text ="Queue stopped"
            else :
                message =f"Encoding queue finished in {self .dashboard .format_time (elapsed )}"
                status_text ="Encoding complete"

            self .dashboard .log_message (message ,"success")
            self .status_label .configure (text =status_text )
        else :
            self .status_label .configure (text ="Ready")

        self .clear_media_cache ()


    def _check_and_finalize (self ):
        """Check if we should finalize and do so if needed."""

        if len (self .worker_threads )==0 and (not self .encoding_active or self .queue_stopped ):
            self ._finalize_encoding_process ()




    def _launch_next_worker (self ):
        """Finds a free worker panel and launches the next available task."""
        if not self .encoding_active or self .queue_stopped :
            return 

        for worker_id ,panel_data in self .dashboard .worker_panels .items ():
            if not panel_data ["busy"]:
                next_file =self .get_next_unprocessed_file ()
                if next_file :

                    panel_data ["busy"]=True 


                    self .dashboard .activate_worker (worker_id ,os .path .basename (next_file ))
                    self .update_queue_item_status (next_file ,"Processing")


                    worker =WorkerThread (next_file ,worker_id ,self .callback_queue ,self .batch_state ,self )
                    worker .start ()
                    self .worker_threads .append (worker )
                else :

                    break 



    def process_callbacks (self ):
        """Process callbacks from worker threads."""
        try :
            if not self .callback_queue .empty ():
                callback =self .callback_queue .get_nowait ()
                worker_id =callback ['worker_id']
                file_path =callback .get ('file_path')



                if callback ['type']in ['worker_complete','worker_skipped','worker_error']:

                    if callback ['type']=='worker_complete':
                        self .dashboard .deactivate_worker (worker_id ,"✅ Success")
                        self .dashboard .log_message (f"SUCCESS: {os .path .basename (file_path )} finished.","success")
                        self .update_queue_item_status (file_path ,"Completed")
                        self .add_file_to_completed (file_path ,'Success')


                    elif callback ['type']=='worker_skipped':
                        self .dashboard .deactivate_worker (worker_id ,"⚠ Skipped")
                        reason =callback .get ('reason','File skipped or failed validation.')

                        self .dashboard .log_message (f"SKIPPED: {os .path .basename (file_path )}","warning")
                        self .update_queue_item_status (file_path ,"Skipped")
                        self .add_file_to_completed (file_path ,'Skipped',skip_reason =reason )

                    elif callback ['type']=='worker_error'and not self .aborting :
                        self .dashboard .deactivate_worker (worker_id ,"❌ Error")
                        error =callback .get ('error','Unknown error')
                        self .dashboard .log_message (f"ERROR on {os .path .basename (file_path )}: {error }","error")
                        self .update_queue_item_status (file_path ,"Failed")
                        self .add_file_to_completed (file_path ,'Failed',skip_reason =error )


                    self .dashboard .worker_panels [worker_id ]["busy"]=False 
                    self .worker_threads =[w for w in self .worker_threads if w .worker_id !=worker_id ]
                    self .dashboard .completed_files +=1
                    

                    if hasattr(self, 'session_encodes_completed'):
                        self.session_encodes_completed += 1
                        
                        # Retrain performance model every 10 encodes
                        if self.session_encodes_completed % 10 == 0 and database_manager:
                            print(f"Triggering mid-session model update after {self.session_encodes_completed} encodes")
                            def retrain_performance():
                                try:
                                    recent_records = database_manager.get_all_performance_records(limit=200)
                                    if self.performance_model and len(recent_records) >= 20:
                                        if self.performance_model.update_model_incrementally(recent_records):
                                            print("Performance model successfully updated mid-session")
                                            self.dashboard.log_message("ML model updated with recent data", "info")
                                except Exception as e:
                                    print(f"Error during mid-session retraining: {e}")
                            
                            # Run in background thread
                            threading.Thread(target=retrain_performance, daemon=True).start()




 

                    if self .encoding_active and not self .queue_stopped :
                        if hasattr (self ,'currently_processing')and file_path in self .currently_processing :
                            self .currently_processing .discard (file_path )
                        self .calculate_ml_based_eta ()

                    if self .dashboard .start_time is not None :
                        elapsed_time =time .time ()-self .dashboard .start_time 
                        self .dashboard .update_stats (
                        self .dashboard .completed_files ,
                        self .batch_state ['total_files'],
                        elapsed_time 
                        )

                    self ._launch_next_worker ()

                    if len (self .worker_threads )==0 and self .get_next_unprocessed_file ()is None :
                        self .encoding_active =False 
                        self .after (100 ,self ._check_and_finalize )


                elif callback ['type']=='worker_log':
                    self .dashboard .log_worker_message (worker_id ,callback ['message'])

                elif callback ['type']=='worker_progress':
                    self .dashboard .update_worker_progress (worker_id ,callback ['progress'])

        except queue .Empty :
            pass 

        if not self .aborting :
            self .after (100 ,self .process_callbacks )




    def add_file_to_completed (self ,file_path :str ,status :str ,skip_reason :str =""):
        """Add file to completed list."""
        try :
            filename =os .path .basename (file_path )


            file_data ={
            'filename':filename ,
            'file_path':file_path ,
            'status':status ,
            'skip_reason':skip_reason ,
            'timestamp':time .time (),
            'initial_size_mb':0 ,
            'output_size_mb':0 ,
            'processing_duration':0 ,
            'final_cq':'N/A',
            'final_score':0 ,
            'ml_accelerated':False 
            }


            if os .path .exists (file_path ):
                file_data ['initial_size_mb']=os .path .getsize (file_path )/(1024 *1024 )


            if status =='Success':
                output_path =self ._find_output_file (file_path )
                if output_path and os .path .exists (output_path ):
                    file_data ['output_size_mb']=os .path .getsize (output_path )/(1024 *1024 )


            if database_manager :
                try :
                    file_hash =database_manager ._get_file_hash (file_path )
                    db_data =database_manager .get_performance_record_by_hash (file_hash )

                    if db_data :
                        file_data ['initial_size_mb'] = db_data.get('size_before_mb', 0)
                        file_data ['final_cq']=db_data .get ('best_cq','N/A')
                        file_data ['final_score']=db_data .get ('final_score',0 )
                        file_data ['processing_duration']=time .time ()-db_data .get ('worker_start_timestamp',time .time ())
                        file_data ['ml_accelerated']=db_data .get ('ml_accelerated',False )

                except Exception as e :
                    print (f"Error getting database data: {e }")


            self ._add_to_completed_tree (file_data )
            self .completed_files .append (file_data )
            self .update_completed_summary ()

        except Exception as e :
            print (f"Error adding completed file: {e }")

    def _add_to_completed_tree (self ,file_data ):
        """Add file data to completed tree view."""
        initial_mb =file_data .get ('initial_size_mb',0 )
        output_mb =file_data .get ('output_size_mb',0 )

        if file_data['status'] == 'Success':
            size_reduction = ((initial_mb - output_mb) / initial_mb * 100) if initial_mb > 0 else 0
            compression_ratio = (initial_mb / output_mb) if output_mb > 0 else 0
            output_size_str = f"{output_mb:.1f} MB"
            size_reduction_str = f"{size_reduction:.1f}%"
            compression_ratio_str = f"{compression_ratio:.1f}x"
            duration_str = self.format_duration(file_data['processing_duration'])
        else:
            output_size_str = "N/A"
            size_reduction_str = "N/A"
            compression_ratio_str = "N/A"
            duration_str = "N/A"

        values =(
        file_data ['status'],
        f"{initial_mb :.1f} MB",
        output_size_str,
        size_reduction_str,
        compression_ratio_str,
        str (file_data ['final_cq']),
        duration_str,
        f"{file_data ['final_score']:.1f}"if file_data ['final_score']>0 else "N/A",
        "Yes"if file_data .get ('ml_accelerated')else "No",
        file_data.get('skip_reason', '') # Add the reason here
        )

        item =self .completed_tree .insert (
        "",
        "end",
        text =file_data ['filename'],
        values =values ,
        tags =(file_data ['status'].lower (),)
        )


        self .completed_tree .tag_configure ("success",foreground ="#2ECC71")
        self .completed_tree .tag_configure ("skipped",foreground ="#F39C12")
        self .completed_tree .tag_configure ("failed",foreground ="#E74C3C")

    def _find_output_file (self ,input_path :str )->str :
        """Find the output file for a given input."""
        if not SETTINGS :
            return ""

        if SETTINGS .output_directory and SETTINGS .output_directory .strip ():
            output_dir =Path (SETTINGS .output_directory )
        else :
            output_dir =Path (input_path ).parent 

        input_stem =Path (input_path ).stem 
        input_suffix =Path (input_path ).suffix 
        output_suffix =SETTINGS .output_suffix 

        expected_output =output_dir /f"{input_stem }{output_suffix }{input_suffix }"

        if expected_output .exists ():
            return str (expected_output )


        for potential_output in output_dir .glob (f"{input_stem }*{output_suffix }*"):
            if potential_output .is_file ():
                return str (potential_output )

        return ""

    def update_completed_summary (self ):
        """Update completed files summary."""
        if not self .completed_files :
            self .summary_label .configure (text ="Summary: 0 files completed | 0 MB saved | 0% average reduction")
            return 

        total_files =len (self .completed_files )
        total_saved_mb =sum (
        (f .get ('initial_size_mb',0 )-f .get ('output_size_mb',0 ))
        for f in self .completed_files if f .get ('status')=='Success'
        )

        successful_files =[f for f in self .completed_files if f .get ('status')=='Success']
        if successful_files :
            avg_reduction =sum (
            ((f .get ('initial_size_mb',0 )-f .get ('output_size_mb',0 ))/f .get ('initial_size_mb',1 )*100 )
            for f in successful_files if f .get ('initial_size_mb',0 )>0 
            )/len (successful_files )
        else :
            avg_reduction =0 

        summary_text =f"Summary: {total_files } files | {total_saved_mb :.1f} MB saved | {avg_reduction :.1f}% avg reduction"
        self .summary_label .configure (text =summary_text )

    def filter_completed_files (self ,filter_value ):
        """Filter completed files by status."""
        for item in self .completed_tree .get_children ():
            self .completed_tree .delete (item )

        for file_data in self .completed_files :
            if filter_value =="All"or file_data ['status']==filter_value :
                self ._add_to_completed_tree (file_data )

    def export_completed_to_csv (self ):
        """Export completed files to CSV."""
        if not self .completed_files :
            messagebox .showinfo ("Export","No completed files to export.")
            return 

        file_path =filedialog .asksaveasfilename (
        defaultextension =".csv",
        filetypes =[("CSV files","*.csv"),("All files","*.*")],
        title ="Export Completed Files"
        )

        if file_path :
            try :
                import csv 
                with open (file_path ,'w',newline ='',encoding ='utf-8')as csvfile :
                    fieldnames =['filename','status','initial_size_mb','output_size_mb',
                    'size_reduction_percent','compression_ratio','final_cq',
                    'processing_duration','final_score','ml_accelerated']

                    writer =csv .DictWriter (csvfile ,fieldnames =fieldnames )
                    writer .writeheader ()

                    for file_data in self .completed_files :
                        row ={key :file_data .get (key ,'')for key in fieldnames }
                        if file_data ['initial_size_mb']>0 and file_data ['output_size_mb']>0 :
                            row ['size_reduction_percent']=((file_data ['initial_size_mb']-file_data ['output_size_mb'])
                            /file_data ['initial_size_mb']*100 )
                            row ['compression_ratio']=file_data ['initial_size_mb']/file_data ['output_size_mb']
                        writer .writerow (row )

                messagebox .showinfo ("Export Complete",f"Data exported to:\n{file_path }")

            except Exception as e :
                messagebox .showerror ("Export Error",f"Failed to export data:\n{str (e )}")

    def clear_completed_list (self ):
        """Clear completed files list."""
        if messagebox .askyesno ("Confirm","Clear all completed files from the list?"):
            self .completed_files .clear ()
            for item in self .completed_tree .get_children ():
                self .completed_tree .delete (item )
            self .update_completed_summary ()

    def format_duration (self ,seconds ):
        """Format duration for display."""
        if seconds <60 :
            return f"{seconds :.0f}s"
        elif seconds <3600 :
            return f"{seconds /60 :.1f}m"
        else :
            return f"{seconds /3600 :.1f}h"

    def clear_media_cache (self ):
        """Clear media info cache."""
        self .media_info_cache .clear ()
        self .analysis_cache .clear ()



    def run_analysis_with_progress (self ,callback_when_done ,callback_on_cancel ):
        """Run file analysis with a queue model and instant process killing."""
        if self .analysis_cancel_event :
            self .analysis_cancel_event .set ()
        self .analysis_cancel_event =threading .Event ()
        self .analysis_cache .clear ()
        self .active_analysis_processes .clear ()

        files_to_analyze =[f for f in self .video_files if f not in self .processed_files ]

        if not files_to_analyze :
            # If pre-filtering removed all files, we just call the completion callback.
            self.status_label.configure(text="No files left after filtering.", text_color="green")
            self .after (0 ,lambda :self ._on_analysis_complete ({}))
            return 

        # MODIFIED: This function now creates the one and only analysis window.
        analysis_dialog = AnalysisProgressDialog(
            self ,len(files_to_analyze ),callback_when_done ,callback_on_cancel
        )

        def analysis_main_thread ():
            file_queue =queue .Queue ()
            results ={}
            lock =threading .Lock ()


            def worker (worker_id :int ):
                while not self .analysis_cancel_event .is_set ():
                    try :
                        file_path =file_queue .get (timeout =0.5 )
                        self .after (0 ,lambda :analysis_dialog .update_file_status (os .path .basename (file_path ),"analyzing"))


                        reason = ""
                        analysis_passed = False
                        
                        media_info = get_media_info(file_path, self.analysis_cancel_event, self.active_analysis_processes, worker_id)

                        if self.analysis_cancel_event.is_set():
                            file_queue.task_done()
                            break

                        if media_info:
                            duration = float(media_info.get('format', {}).get('duration', 0))
                            sample_points, complexity_data = get_final_sample_points(
                                file_path, 0, lambda id, msg: None,
                                {'video_duration_seconds': duration}
                            )
                            if sample_points:
                                complexity_data['sample_points'] = sample_points
                                result_data = {'media_info': media_info, 'complexity_data': complexity_data}
                                with lock:
                                    results[file_path] = result_data
                                self.after(0, lambda p=file_path, r=result_data: analysis_dialog.add_analysis_result(p, r))
                                self.after(0, lambda: analysis_dialog.update_file_status(os.path.basename(file_path), "completed"))
                                analysis_passed = True
                            else:
                                reason = "No sample points"
                        else:
                            if not self.analysis_cancel_event.is_set():
                                reason = "Could not read media info"

                        if not analysis_passed and reason:
                            with lock:
                                results[file_path] = {'status': 'analysis_failed', 'reason': reason}
                            self.after(0, lambda: analysis_dialog.update_file_status(os.path.basename(file_path), "failed", reason))
                        # --- End of Replacement ---

                        file_queue .task_done ()

                    except queue .Empty :
                        break 
                    except Exception as e :
                        print (f"Error in analysis worker {worker_id }: {e }")
                        if file_queue .unfinished_tasks >0 :
                            file_queue .task_done ()


            num_workers =min (4 ,os .cpu_count ()or 1 )
            threads =[threading .Thread (target =worker ,args =(i ,),daemon =True )for i in range (num_workers )]
            for t in threads :
                t .start ()




            for file in files_to_analyze :
                if self .analysis_cancel_event .is_set ():
                    break 
                file_queue .put (file )



            file_queue .join ()
            for t in threads :
                t .join ()


            if not self .analysis_cancel_event .is_set ():
                if analysis_dialog .winfo_exists ():
                    self .after (0 ,lambda :callback_when_done (results ))
            else :
                if analysis_dialog .winfo_exists ():

                    pass 

        threading .Thread (target =analysis_main_thread ,daemon =True ).start ()




    def _analysis_cancelled (self ):
        """Handle cancelled analysis."""
        if hasattr (self ,'analysis_window')and self .analysis_window .winfo_exists ():
            self .analysis_window .destroy ()


        self .encoding_active =False 
        self .start_button .configure (state ="normal")
        self .stop_queue_button .configure (state ="disabled")
        self .abort_button .configure (state ="disabled")
        self .settings_button .configure (state ="normal")
        self .dashboard .log_message ("Analysis cancelled by user","warning")
        self .status_label .configure (text ="Analysis cancelled")





    def on_closing (self ):
        """Handle window closing."""
        if self .encoding_active :
            if messagebox .askyesno ("Confirm","Encoding in progress. Abort and exit?"):
                self .abort_process ()
            else :
                return 

        self .destroy ()


def main ():
    """Point d'entrée principal."""
    app =AutoVMAFEncoderGUI ()
    app .mainloop ()

if __name__ =="__main__":
    main ()






