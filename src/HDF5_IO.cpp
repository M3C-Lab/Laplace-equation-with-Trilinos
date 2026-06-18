#include "HDF5_IO.hpp"

#include <stdexcept>

namespace laplace {

namespace {

void checkStatus(const herr_t status, const char* message) {
  if (status < 0) {
    throw std::runtime_error(message);
  }
}

hid_t openGroup(const hid_t fileId, const std::string& path) {
  const hid_t groupId = H5Gopen2(fileId, path.c_str(), H5P_DEFAULT);
  if (groupId < 0) {
    throw std::runtime_error("Failed to open HDF5 group: " + path);
  }
  return groupId;
}

template <typename T>
void writeScalarImpl(
  const hid_t location,
  const std::string& name,
  const hid_t nativeType,
  const T& value) {
  const hsize_t dims[1] = {1};
  const hid_t dataspace = H5Screate_simple(1, dims, nullptr);
  const hid_t dataset = H5Dcreate2(
    location,
    name.c_str(),
    nativeType,
    dataspace,
    H5P_DEFAULT,
    H5P_DEFAULT,
    H5P_DEFAULT);
  checkStatus(H5Dwrite(dataset, nativeType, H5S_ALL, H5S_ALL, H5P_DEFAULT, &value), "Failed to write HDF5 scalar.");
  H5Dclose(dataset);
  H5Sclose(dataspace);
}

template <typename T>
void writeVectorImpl(
  const hid_t location,
  const std::string& name,
  const hid_t nativeType,
  const std::vector<T>& values) {
  const hsize_t dims[1] = {static_cast<hsize_t>(values.size())};
  const hid_t dataspace = H5Screate_simple(1, dims, nullptr);
  const hid_t dataset = H5Dcreate2(
    location,
    name.c_str(),
    nativeType,
    dataspace,
    H5P_DEFAULT,
    H5P_DEFAULT,
    H5P_DEFAULT);
  const void* dataPtr = values.empty() ? nullptr : static_cast<const void*>(values.data());
  checkStatus(H5Dwrite(dataset, nativeType, H5S_ALL, H5S_ALL, H5P_DEFAULT, dataPtr), "Failed to write HDF5 vector.");
  H5Dclose(dataset);
  H5Sclose(dataspace);
}

template <typename T>
std::vector<T> readVectorImpl(
  const hid_t fileId,
  const std::string& groupPath,
  const std::string& name,
  const hid_t nativeType) {
  const hid_t groupId = openGroup(fileId, groupPath);
  const hid_t dataset = H5Dopen2(groupId, name.c_str(), H5P_DEFAULT);
  const hid_t dataspace = H5Dget_space(dataset);

  hsize_t dims[2] = {0, 0};
  checkStatus(H5Sget_simple_extent_dims(dataspace, dims, nullptr), "Failed to read HDF5 vector dims.");
  std::vector<T> values(static_cast<std::size_t>(dims[0]));
  void* dataPtr = values.empty() ? nullptr : static_cast<void*>(values.data());
  checkStatus(H5Dread(dataset, nativeType, H5S_ALL, H5S_ALL, H5P_DEFAULT, dataPtr), "Failed to read HDF5 vector.");

  H5Sclose(dataspace);
  H5Dclose(dataset);
  H5Gclose(groupId);
  return values;
}

template <typename T>
T readScalarImpl(
  const hid_t fileId,
  const std::string& groupPath,
  const std::string& name,
  const hid_t nativeType) {
  const hid_t groupId = openGroup(fileId, groupPath);
  const hid_t dataset = H5Dopen2(groupId, name.c_str(), H5P_DEFAULT);
  T value{};
  checkStatus(H5Dread(dataset, nativeType, H5S_ALL, H5S_ALL, H5P_DEFAULT, &value), "Failed to read HDF5 scalar.");
  H5Dclose(dataset);
  H5Gclose(groupId);
  return value;
}

}  // namespace

HDF5Writer::HDF5Writer(const std::string& fileName) {
  fileId_ = H5Fcreate(fileName.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
  if (fileId_ < 0) {
    throw std::runtime_error("Failed to create HDF5 file: " + fileName);
  }
}

HDF5Writer::~HDF5Writer() {
  if (fileId_ >= 0) {
    H5Fclose(fileId_);
  }
}

hid_t HDF5Writer::createGroup(const std::string& path) const {
  const hid_t groupId = H5Gcreate2(fileId_, path.c_str(), H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
  if (groupId < 0) {
    throw std::runtime_error("Failed to create HDF5 group: " + path);
  }
  return groupId;
}

void HDF5Writer::closeGroup(const hid_t groupId) const {
  H5Gclose(groupId);
}

void HDF5Writer::writeIntScalar(const hid_t location, const std::string& name, const int value) const {
  writeScalarImpl(location, name, H5T_NATIVE_INT, value);
}

void HDF5Writer::writeDoubleScalar(const hid_t location, const std::string& name, const double value) const {
  writeScalarImpl(location, name, H5T_NATIVE_DOUBLE, value);
}

void HDF5Writer::writeIntVector(const hid_t location, const std::string& name, const std::vector<int>& values) const {
  writeVectorImpl(location, name, H5T_NATIVE_INT, values);
}

void HDF5Writer::writeDoubleVector(const hid_t location, const std::string& name, const std::vector<double>& values) const {
  writeVectorImpl(location, name, H5T_NATIVE_DOUBLE, values);
}

void HDF5Writer::writeIntMatrix(
  const hid_t location,
  const std::string& name,
  const std::vector<int>& values,
  const int rows,
  const int cols) const {
  const hsize_t dims[2] = {static_cast<hsize_t>(rows), static_cast<hsize_t>(cols)};
  const hid_t dataspace = H5Screate_simple(2, dims, nullptr);
  const hid_t dataset = H5Dcreate2(
    location,
    name.c_str(),
    H5T_NATIVE_INT,
    dataspace,
    H5P_DEFAULT,
    H5P_DEFAULT,
    H5P_DEFAULT);
  const void* dataPtr = values.empty() ? nullptr : static_cast<const void*>(values.data());
  checkStatus(H5Dwrite(dataset, H5T_NATIVE_INT, H5S_ALL, H5S_ALL, H5P_DEFAULT, dataPtr), "Failed to write HDF5 matrix.");
  H5Dclose(dataset);
  H5Sclose(dataspace);
}

HDF5Reader::HDF5Reader(const std::string& fileName) {
  fileId_ = H5Fopen(fileName.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
  if (fileId_ < 0) {
    throw std::runtime_error("Failed to open HDF5 file: " + fileName);
  }
}

HDF5Reader::~HDF5Reader() {
  if (fileId_ >= 0) {
    H5Fclose(fileId_);
  }
}

int HDF5Reader::readIntScalar(const std::string& groupPath, const std::string& name) const {
  return readScalarImpl<int>(fileId_, groupPath, name, H5T_NATIVE_INT);
}

double HDF5Reader::readDoubleScalar(const std::string& groupPath, const std::string& name) const {
  return readScalarImpl<double>(fileId_, groupPath, name, H5T_NATIVE_DOUBLE);
}

std::vector<int> HDF5Reader::readIntVector(const std::string& groupPath, const std::string& name) const {
  return readVectorImpl<int>(fileId_, groupPath, name, H5T_NATIVE_INT);
}

std::vector<double> HDF5Reader::readDoubleVector(const std::string& groupPath, const std::string& name) const {
  return readVectorImpl<double>(fileId_, groupPath, name, H5T_NATIVE_DOUBLE);
}

std::vector<int> HDF5Reader::readIntMatrix(
  const std::string& groupPath,
  const std::string& name,
  int& rows,
  int& cols) const {
  const hid_t groupId = openGroup(fileId_, groupPath);
  const hid_t dataset = H5Dopen2(groupId, name.c_str(), H5P_DEFAULT);
  const hid_t dataspace = H5Dget_space(dataset);
  hsize_t dims[2] = {0, 0};
  checkStatus(H5Sget_simple_extent_dims(dataspace, dims, nullptr), "Failed to read HDF5 matrix dims.");
  rows = static_cast<int>(dims[0]);
  cols = static_cast<int>(dims[1]);
  std::vector<int> values(static_cast<std::size_t>(rows * cols));
  void* dataPtr = values.empty() ? nullptr : static_cast<void*>(values.data());
  checkStatus(H5Dread(dataset, H5T_NATIVE_INT, H5S_ALL, H5S_ALL, H5P_DEFAULT, dataPtr), "Failed to read HDF5 matrix.");
  H5Sclose(dataspace);
  H5Dclose(dataset);
  H5Gclose(groupId);
  return values;
}

}  // namespace laplace
