from mpi4py import MPI
import cv2
import threading
import queue
import numpy as np


class WorkerThread(threading.Thread):
    def __init__(self, task_queue, output, rank=0):
        super().__init__()
        self.task_queue = task_queue
        self.output = output
        self.rank = rank

    def run(self):
        while True:
            task = self.task_queue.get()
            if task is None:
                print(f"thread {self.name} ready.")
                break
            image, operation = task
            result = self.process_image(image, operation)
            print(f"thread {self.name}  processing {image} with {operation}.")
            self.output.put(result)

    def process_image(self, image, operation):
        img = cv2.imread(image, cv2.IMREAD_COLOR)
        if operation == 'edge_detection':
            result = cv2.Canny(img, 100, 200)
        elif operation == 'color_inversion':
            result = cv2.bitwise_not(img)
        elif operation == 'custom_processing':
            result = self.custom_processing(img)
        return result

    def custom_processing(self, img):
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        v += 50
        final_hsv = cv2.merge((h, s, v))
        result = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
        return result

def main():
 
    rank = MPI.COMM_WORLD.Get_rank()
    size = MPI.COMM_WORLD.Get_size()
    task_queue = [("img1.jpg", "edge_detection"), ("img1.jpg", "color_inversion"),
                  ("img1.jpg", "custom_processing"), ("img2.jpg", "edge_detection"),
                  ("img2.jpg", "color_inversion"),
                  ("img2.jpg", "custom_processing")]

    if size == 1:
        local_task_queue = queue.Queue()
        local_output = queue.Queue()
        for task in task_queue:
            local_task_queue.put(task)

        local_num_threads = 2

        for _ in range(local_num_threads):
            local_task_queue.put(None)

        threads = []
        for _ in range(local_num_threads):
            thread = WorkerThread(local_task_queue, local_output)
            threads.append(thread)
            thread.start()

        for thread in threads:
            thread.join()

        i = 0
        while not local_output.empty():
            result = local_output.get()
            filename = f"modified{i}.jpg"
            cv2.imwrite(filename, result)
            print(f"Saved {filename}")
            i += 1

        
    else:
        if rank == 0:
            task_queue_size = len(task_queue)
            tasks_per_process = task_queue_size // (size - 1)
            remainder = task_queue_size % (size - 1)

            start_index = 0
            for dest in range(1, size):
                if dest <= remainder:
                    num_of_task_for_current_process = tasks_per_process + 1;
                else:
                    num_of_task_for_current_process = tasks_per_process;

                end_index = start_index + num_of_task_for_current_process;
                subtasks = task_queue[start_index:end_index]
                print(f"Sending tasks {subtasks} to process {dest}")
                MPI.COMM_WORLD.send(subtasks, dest=dest)
                start_index = end_index

            results = []
            for _ in range(size - 1):
                received_data = MPI.COMM_WORLD.recv(source=MPI.ANY_SOURCE)
                if isinstance(received_data, list):
                    results.extend(received_data)
                else:
                    results.append(received_data)

            for i, result in enumerate(results):
                filename = f"modified{i}.jpg"
                cv2.imwrite(filename, result)
                print(f"Saved {filename}")

            

        else:
            print(f"Process {rank} in the blocked worker ")
            task_queue = queue.Queue()
            output = queue.Queue()
            tasks = MPI.COMM_WORLD.recv(source=0)
            print(f"Process {rank} received task: {tasks}")
            for task in tasks:
                task_queue.put(task)

            num_threads = 1
            for _ in range(num_threads):
                task_queue.put(None)

            threads = []
            for _ in range(num_threads):
                thread = WorkerThread(task_queue, output, rank)
                threads.append(thread)
                thread.start()

            for thread in threads:
                thread.join()

            results_to_send = []
            while not output.empty():
                result = output.get()
                results_to_send.append(result)

            MPI.COMM_WORLD.send(results_to_send, dest=0)

if __name__ == "__main__":
    main()
