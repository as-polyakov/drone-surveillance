# drone-surveillance


enable wired connection:
```aiignore
sudo nmcli con mod "Wired connection 1" ipv4.method manual
sudo nmcli con mod "Wired connection 1" ipv4.addresses 192.168.1.2/24
sudo nmcli con up "Wired connection 1"
```