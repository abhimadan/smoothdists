#pragma once

struct Image {
  unsigned char* const R;
  unsigned char* const G;
  unsigned char* const B;
  int sizex, sizey;

  Image() : R(nullptr), G(nullptr), B(nullptr), sizex(0), sizey(0) {}
  Image(unsigned char* R, unsigned char* G, unsigned char* B, int sizex,
        int sizey)
      : R(R), G(G), B(B), sizex(sizex), sizey(sizey) {}

  // Can also check other sizes and pointers but this is faster
  bool isEmpty() const { return sizex == 0; }

  unsigned char red(int x, int y) const {
    return R[y*sizex + x];
  }
  unsigned char& red(int x, int y) {
    return R[y*sizex + x];
  }
  unsigned char green(int x, int y) const {
    return G[y*sizex + x];
  }
  unsigned char& green(int x, int y) {
    return G[y*sizex + x];
  }
  unsigned char blue(int x, int y) const {
    return B[y*sizex + x];
  }
  unsigned char& blue(int x, int y) {
    return B[y*sizex + x];
  }
};
