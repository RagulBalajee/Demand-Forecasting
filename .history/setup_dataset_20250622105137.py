#!/usr/bin/env python3
"""
Setup script to organize the Rossmann Store Sales dataset
"""

import os
import shutil
import sys
from pathlib import Path

def setup_rossmann_dataset():
    """Setup the Rossmann dataset in the correct location"""
    
    print("🛍️ Rossmann Store Sales Dataset Setup")
    print("=" * 50)
    
    # Check if dataset is already in the correct location
    target_dir = Path("data/rossmann-store-sales")
    
    if target_dir.exists():
        print(f"✅ Dataset already exists in {target_dir}")
        print("Files found:")
        for file in target_dir.glob("*.csv"):
            print(f"   📄 {file.name}")
        return True
    
    # Look for the dataset in common locations
    possible_locations = [
        Path("rossmann-store-sales"),
        Path("../rossmann-store-sales"),
        Path("../../rossmann-store-sales"),
        Path.home() / "Downloads" / "rossmann-store-sales",
        Path.home() / "Desktop" / "rossmann-store-sales"
    ]
    
    source_dir = None
    for loc in possible_locations:
        if loc.exists() and loc.is_dir():
            # Check if it contains the expected files
            csv_files = list(loc.glob("*.csv"))
            if len(csv_files) >= 3:  # Should have at least train.csv, store.csv, test.csv
                source_dir = loc
                break
    
    if source_dir is None:
        print("❌ Rossmann dataset not found in common locations.")
        print("\n📁 Please place your dataset in one of these locations:")
        print("   - ./rossmann-store-sales/")
        print("   - ../rossmann-store-sales/")
        print("   - ~/Downloads/rossmann-store-sales/")
        print("   - ~/Desktop/rossmann-store-sales/")
        print("\n📋 Expected files:")
        print("   - train.csv")
        print("   - test.csv")
        print("   - store.csv")
        print("   - sample_submission.csv")
        return False
    
    print(f"📁 Found dataset in: {source_dir}")
    print("📄 Files found:")
    for file in source_dir.glob("*.csv"):
        print(f"   - {file.name}")
    
    # Create target directory
    target_dir.parent.mkdir(exist_ok=True)
    
    # Copy dataset
    print(f"\n📋 Copying dataset to: {target_dir}")
    try:
        shutil.copytree(source_dir, target_dir)
        print("✅ Dataset copied successfully!")
        
        # Verify the copy
        print("\n📊 Verifying dataset:")
        for file in target_dir.glob("*.csv"):
            size = file.stat().st_size / 1024  # Size in KB
            print(f"   ✅ {file.name} ({size:.1f} KB)")
        
        return True
        
    except Exception as e:
        print(f"❌ Error copying dataset: {e}")
        return False

def test_dataset_loading():
    """Test if the dataset can be loaded correctly"""
    
    print("\n🧪 Testing dataset loading...")
    
    try:
        from data.loader import DataLoader
        
        loader = DataLoader()
        sales_df, store_df = loader.load_rossmann_data()
        
        print(f"✅ Successfully loaded dataset!")
        print(f"   📊 Sales records: {len(sales_df):,}")
        print(f"   🏪 Stores: {sales_df['Store'].nunique()}")
        print(f"   📅 Date range: {sales_df['Date'].min()} to {sales_df['Date'].max()}")
        print(f"   💰 Total sales: {sales_df['Sales'].sum():,}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error loading dataset: {e}")
        return False

def main():
    """Main setup function"""
    
    print("🚀 Setting up Rossmann Store Sales Dataset")
    print("=" * 60)
    
    # Setup dataset
    if not setup_rossmann_dataset():
        print("\n❌ Setup failed. Please ensure the dataset is in the correct location.")
        sys.exit(1)
    
    # Test loading
    if not test_dataset_loading():
        print("\n❌ Dataset loading failed. Please check the file format.")
        sys.exit(1)
    
    print("\n" + "=" * 60)
    print("🎉 Dataset setup completed successfully!")
    print("\n🚀 You can now:")
    print("   1. Run demo with real data: python demo_rossmann.py")
    print("   2. Start web app: streamlit run app.py")
    print("   3. Use 'Rossmann Dataset' option in the web app")
    print("=" * 60)

if __name__ == "__main__":
    main() 