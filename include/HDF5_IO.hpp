#pragma once

#include <hdf5.h>

#include <string>
#include <vector>

namespace laplace {

class HDF5Writer {
 public:
  explicit HDF5Writer(const std::string& fileName);
  ~HDF5Writer();

  hid_t fileId() const { return fileId_; }

  hid_t createGroup(const std::string& path) const;
  void closeGroup(hid_t groupId) const;

  void writeIntScalar(hid_t location, const std::string& name, int value) const;
  void writeDoubleScalar(hid_t location, const std::string& name, double value) const;
  void writeIntVector(hid_t location, const std::string& name, const std::vector<int>& values) const;
  void writeDoubleVector(hid_t location, const std::string& name, const std::vector<double>& values) const;
  void writeIntMatrix(
    hid_t location,
    const std::string& name,
    const std::vector<int>& values,
    int rows,
    int cols) const;

 private:
  hid_t fileId_ = -1;
};

class HDF5Reader {
 public:
  explicit HDF5Reader(const std::string& fileName);
  ~HDF5Reader();

  int readIntScalar(const std::string& groupPath, const std::string& name) const;
  double readDoubleScalar(const std::string& groupPath, const std::string& name) const;
  std::vector<int> readIntVector(const std::string& groupPath, const std::string& name) const;
  std::vector<double> readDoubleVector(const std::string& groupPath, const std::string& name) const;
  std::vector<int> readIntMatrix(
    const std::string& groupPath,
    const std::string& name,
    int& rows,
    int& cols) const;

 private:
  hid_t fileId_ = -1;
};

}  // namespace laplace
