import numpy as np
import cv2
import hashlib
import os

class SectionsSplitter():
    def __create_files(self, source):
        files = []
        index_of_dot = source.index('.')
        file_name_without_extension = source[:index_of_dot]
        # b_source = bytearray()
        # b_source.extend(source)
        # file_hash = hashlib.md5(b_source).hexdigest()

        for i in xrange(self.num_of_splits * self.num_of_splits):
            filename = file_name_without_extension + '_' + str(i) + '.avi'
            files.append(filename)
        return files

    def __create_writers(self, files):
        writers = []

        fourcc = cv2.VideoWriter_fourcc(*'XVID')

        for filename in files:
            writer = cv2.VideoWriter(filename, fourcc, 29.0, (self.split_width_size, self.split_height_size))
            writers.append(writer)
        return writers

    def __release_writers(self, writers):
        for writer in writers:
            writer.release()

    def __prepare_to_split(self, file_to_split, num_of_splits):
        self.num_of_splits = num_of_splits
        self.cap = cv2.VideoCapture(file_to_split)
        if not self.cap.isOpened():
            raise Exception("Failed to open file.")

        self.width = self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)   # float
        self.height = self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT) # float

        self.split_width_size = int(self.width / num_of_splits)
        self.split_height_size = int(self.height / num_of_splits)


    def split(self, file_to_split, num_of_splits=4):
        self.__prepare_to_split(file_to_split, num_of_splits)

        files = self.__create_files(file_to_split)

        writers = self.__create_writers(files)

        while(self.cap.isOpened()):
            ret, frame = self.cap.read()

            if ret==True:
                for i in xrange(self.num_of_splits):
                    for j in xrange(self.num_of_splits):
                        relevant_frame = frame[i*self.split_height_size:(i+1)*self.split_height_size,j*self.split_width_size:(j+1)*self.split_width_size]       

                        # print(i*SPLIT_TO + j)
                        writers[i*self.num_of_splits + j].write(relevant_frame)
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            else:
                break

        self.__release_writers(writers)

        self.cap.release()
        cv2.destroyAllWindows()

        new_file_name = file_to_split[:file_to_split.index('.')] + '_' + str(self.num_of_splits * num_of_splits) + '.avi'
        os.rename(file_to_split, new_file_name)
        files.append(new_file_name)

        return files

if __name__ == "__main__":
    section_splitter =  SectionsSplitter()

    print(section_splitter.split(r"C:\videos\1\1\1_40.avi"))