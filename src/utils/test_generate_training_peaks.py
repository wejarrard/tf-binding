import unittest
import pandas as pd
import numpy as np
from unittest.mock import patch, Mock, mock_open
from io import StringIO
import tempfile
import os

# Import your module here
from generate_training_peaks import *

class TestDataProcessingFunctions(unittest.TestCase):
    def setUp(self):
        """Set up test data"""
        # Sample DataFrame for testing
        self.test_df = pd.DataFrame({
            'chr': ['chr1', 'chr2', 'chrX', 'chr10', 'chrM'],
            'start': [100, 200, 300, 400, 500],
            'end': [1100, 2200, 3300, 4400, 5500],
            'count': [5, 15, 25, 35, 45]
        })

    def test_filter_peak_lengths(self):
        """Test filtering peaks based on length"""
        # Test with default threshold
        result = filter_peak_lengths(self.test_df.copy())
        self.assertEqual(len(result), 4)  # chrM row should be filtered out

        # Test with custom threshold
        result = filter_peak_lengths(self.test_df.copy(), threshold=2000)
        self.assertEqual(len(result), 2)  # Only first two rows should remain

    def test_filter_chromosomes(self):
        """Test chromosome filtering"""
        result = filter_chromosomes(self.test_df.copy())
        self.assertEqual(len(result), 4)  # chrM should be filtered out
        self.assertTrue('chrM' not in result['chr'].values)
        self.assertTrue(all(chr in result['chr'].values for chr in ['chr1', 'chr2', 'chrX', 'chr10']))

    def test_label_peaks(self):
        """Test peak labeling"""
        # Test with drop_rows=True
        result = label_peaks(self.test_df.copy(), balance=False, drop_rows=True)
        self.assertTrue('label' in result.columns)
        self.assertTrue(all(result['label'] == 1))

        # Test with drop_rows=False
        result = label_peaks(self.test_df.copy(), balance=False, drop_rows=False)
        self.assertTrue('label' in result.columns)
        self.assertTrue(any(result['label'] == 0))
        self.assertTrue(any(result['label'] == 1))

    def test_balance_labels(self):
        """Test label balancing"""
        # Create unbalanced dataset
        unbalanced_df = pd.DataFrame({
            'chr': ['chr1']*7 + ['chr2']*3,
            'start': range(10),
            'end': range(10, 20),
            'label': [1]*7 + [0]*3
        })
        
        result = balance_labels(unbalanced_df)
        label_counts = result['label'].value_counts()
        self.assertEqual(label_counts[0], label_counts[1])
        self.assertEqual(len(result), 6)  # 3 samples of each class

class TestBedFileOperations(unittest.TestCase):
    @patch('subprocess.run')
    def test_subtract_bed_files(self, mock_run):
        """Test bed file subtraction"""
        df1 = pd.DataFrame({
            'chr': ['chr1', 'chr2'],
            'start': [100, 200],
            'end': [150, 250]
        })
        
        df2 = pd.DataFrame({
            'chr': ['chr1'],
            'start': [120],
            'end': [130]
        })

        # Mock subprocess.run to simulate bedtools behavior
        mock_run.return_value = Mock(returncode=0)
        
        # Mock the file operations
        with patch('tempfile.NamedTemporaryFile') as mock_temp:
            mock_temp.return_value.__enter__.return_value.name = 'temp_file'
            result = subtract_bed_files(df1, df2)
            
            self.assertTrue(isinstance(result, pd.DataFrame))
            mock_run.assert_called_once()

    @patch('subprocess.run')
    def test_intersect_bed_files(self, mock_run):
        """Test bed file intersection"""
        df1 = pd.DataFrame({
            'chr': ['chr1', 'chr2'],
            'start': [100, 200],
            'end': [150, 250]
        })
        
        df2 = pd.DataFrame({
            'chr': ['chr1'],
            'start': [120],
            'end': [130]
        })

        # Mock subprocess.run to simulate bedtools behavior
        mock_run.return_value = Mock(returncode=0)
        
        # Mock the file operations
        with patch('tempfile.NamedTemporaryFile') as mock_temp:
            mock_temp.return_value.__enter__.return_value.name = 'temp_file'
            result = intersect_bed_files(df1, df2)
            
            self.assertTrue(isinstance(result, pd.DataFrame))
            mock_run.assert_called_once()

class TestArgumentParsing(unittest.TestCase):
    def test_parse_arguments_required_args(self):
        """Test argument parsing with required arguments"""
        with patch('sys.argv', ['script.py', '--tf', 'TEST_TF']):
            args = parse_arguments()
            self.assertEqual(args.tf, 'TEST_TF')
            self.assertFalse(args.balance)
            self.assertFalse(args.enhancer_promotor_only)

    def test_parse_arguments_validation_group(self):
        """Test mutually exclusive validation group"""
        with patch('sys.argv', [
            'script.py', 
            '--tf', 'TEST_TF',
            '--validation_cell_lines', 'cell1', 'cell2'
        ]):
            args = parse_arguments()
            self.assertEqual(args.validation_cell_lines, ['cell1', 'cell2'])
            self.assertIsNone(args.validation_chromosomes)

    def test_parse_arguments_invalid_combination(self):
        """Test invalid argument combinations"""
        with patch('sys.argv', [
            'script.py',
            '--chip_provided',
            '--tf', 'TEST_TF'
        ]):
            with self.assertRaises(SystemExit):
                parse_arguments()

def run_tests():
    unittest.main()

if __name__ == '__main__':
    run_tests()