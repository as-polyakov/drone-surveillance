# Variables
PACKAGE_NAME = drone-surveillance
VERSION = 1.0.0
ARCH = arm64
DEB_PACKAGE = $(PACKAGE_NAME)_$(VERSION)_$(ARCH).deb
PACKAGE_DIR = package
INSTALL_DIR = $(PACKAGE_DIR)/usr/local/lib/drone-surveillance
SYSTEMD_DIR = $(PACKAGE_DIR)/etc/systemd/system
DEBIAN_DIR = $(PACKAGE_DIR)/DEBIAN

# Source directories (relative to debian-pkg directory)
SRC_DIR = ../src
ROOT_DIR = ..

# Source files
DEBIAN_FILES = control postinst prerm
SERVICE_FILE = drone-surveillance.service

.PHONY: all clean build install

all: build

# Create directory structure
$(PACKAGE_DIR):
	mkdir -p $(INSTALL_DIR)
	mkdir -p $(SYSTEMD_DIR)
	mkdir -p $(DEBIAN_DIR)

# Copy source directory
$(INSTALL_DIR)/src: $(SRC_DIR)
	cp -r $< $@

# Copy requirements.txt
$(INSTALL_DIR)/requirements.txt: $(ROOT_DIR)/requirements.txt
	cp $< $@

# Copy systemd service
$(SYSTEMD_DIR)/$(SERVICE_FILE): etc/systemd/system/$(SERVICE_FILE)
	cp $< $@

# Copy DEBIAN control files
$(DEBIAN_DIR)/%: DEBIAN/%
	cp $< $@
	chmod 755 $@

# Build the package
build: $(PACKAGE_DIR) \
       $(INSTALL_DIR)/src \
       $(INSTALL_DIR)/requirements.txt \
       $(SYSTEMD_DIR)/$(SERVICE_FILE) \
       $(addprefix $(DEBIAN_DIR)/,$(DEBIAN_FILES))
	dpkg-deb --build $(PACKAGE_DIR) $(DEB_PACKAGE)
	@echo "Package built: $(DEB_PACKAGE)"

# Clean build artifacts
clean:
	rm -rf $(PACKAGE_DIR)
	rm -f $(DEB_PACKAGE)

# Install the package (requires sudo)
install: build
	sudo dpkg -i $(DEB_PACKAGE)

# Help target
help:
	@echo "Available targets:"
	@echo "  all      - Build the package (default)"
	@echo "  build    - Build the package"
	@echo "  clean    - Remove build artifacts"
	@echo "  install  - Install the package (requires sudo)"
	@echo "  help     - Show this help message" 