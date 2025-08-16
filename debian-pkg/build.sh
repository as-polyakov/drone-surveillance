#!/bin/bash
set -e

# Create package directory structure
mkdir -p package/usr/local/lib/drone-surveillance
mkdir -p package/etc/systemd/system

# Copy application files
cp video.py package/usr/local/lib/drone-surveillance/
cp requirements.txt package/usr/local/lib/drone-surveillance/
cp yolov8n-fire-best.pt package/usr/local/lib/drone-surveillance/

# Copy systemd service
cp etc/systemd/system/drone-surveillance.service package/etc/systemd/system/

# Copy DEBIAN control files
cp DEBIAN/control package/DEBIAN/
cp DEBIAN/postinst package/DEBIAN/
cp DEBIAN/prerm package/DEBIAN/

# Set permissions
chmod 755 package/DEBIAN/postinst
chmod 755 package/DEBIAN/prerm

# Build the package
dpkg-deb --build package drone-surveillance_1.0.0_armhf.deb

echo "Package built: drone-surveillance_1.0.0_armhf.deb" 