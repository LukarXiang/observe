from .publication import BatchPublisher


def publish_batch(root, batch_id, tables):
    publisher = BatchPublisher(root); publisher.write_batch(batch_id, tables); return publisher.publish(batch_id)
