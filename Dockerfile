FROM archlinux

# Needed to get around PGP errors
#   e.g. "key "991F6E3F0765CF6295888586139B09DA5BF0D338" is unknown"
RUN pacman-key --init
RUN pacman-key --populate archlinux
RUN pacman-key --refresh-keys
RUN pacman -Sy --noconfirm archlinux-keyring

# Need a new user to build things (makepkg should not be run as root)
RUN useradd -m build && \
    pacman -Syu --noconfirm --needed && \
    pacman -Sy --noconfirm --needed archlinux-keyring && \
    pacman -Sy --noconfirm --needed make python2 

# Needed later for yay
RUN pacman -Sy --noconfirm --needed git base-devel go

# Shouldn't have to make this directory...?
RUN mkdir -p /etc/sudoers.d && \
    touch /etc/sudoers.d/build && \
    echo "build ALL=(ALL) NOPASSWD: ALL" > /etc/sudoers.d/build
# Switch to build user
USER build
WORKDIR /home/build
# Get yay for AUR packages
RUN git clone https://aur.archlinux.org/yay.git && \
    cd yay && \
    makepkg --noconfirm -si --needed

RUN yay -Sy --noconfirm --needed micromamba-bin qt4

RUN micromamba create -n spectro_env -c conda-forge python=2.7 pyqt=4.11.4 matplotlib numpy scipy xlsxwriter=1.1.5 lmfit=0.9.11

############################ | Package Specific | ############################# 
RUN mkdir /script
WORKDIR /script
COPY Spectro.py /script
CMD ["/usr/bin/python2", "Spectro.py"]
