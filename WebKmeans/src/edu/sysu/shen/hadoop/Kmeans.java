package edu.sysu.shen.hadoop;

import java.io.IOException;
import java.text.DecimalFormat;
import java.util.HashMap;
import java.util.Map;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.conf.Configured;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.SequenceFile;
import org.apache.hadoop.io.SequenceFile.Reader;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.util.Tool;
import org.apache.hadoop.util.ToolRunner;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.input.SequenceFileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.hadoop.mapreduce.lib.output.SequenceFileOutputFormat;

/**
 * Kmeans 聚类 输出key:文档编号 value:index:tfidf1/index:tfidf2/.../ 输出key:文档所属中心点
 * value: 文档 使用hadoop1.0.4
 * 
 * @author 申国骏
 */
public class Kmeans extends Configured implements Tool {

	// 主目录
	private static final String BASE_PATH = "/usr/shen/chinesewebkmeans";
	private static final String CEN_PATH = "/usr/shen/chinesewebkmeans/center/centroid.list";
	private static final String DICT_PATH = "/usr/shen/chinesewebkmeans/dict/dict.list";

	public static class KmeansMapper extends
			Mapper<LongWritable, Text, IntWritable, Text> {

		private static Map<Integer, Map<Long, Double>> centers = new HashMap<Integer, Map<Long, Double>>();
		private static Map<String, Long> dictWords = new HashMap<String, Long>();
		private IntWritable classCenter;

		/**
		 * 将文档中心从hdfs中加载至内存
		 */
		@Override
		protected void setup(Context context) throws IOException,
				InterruptedException {
			Configuration conf = context.getConfiguration();
			Path centroids = new Path(conf.get("CENPATH"));
			Path dictPath = new Path(conf.get("DICTPATH"));
			FileSystem fs = FileSystem.get(conf);

			SequenceFile.Reader reader = new SequenceFile.Reader(fs, centroids,
					conf);
			IntWritable key = new IntWritable();
			Text value = new Text();
			while (reader.next(key, value)) {
				Map<Long, Double> tfidfAndIndex = new HashMap<Long, Double>();
				String[] iat = value.toString().split("/");
				for (String string : iat) {
					tfidfAndIndex.put(Long.parseLong(string.split(":")[0]),
							Double.parseDouble(string.split(":")[1]));
				}
				centers.put(key.get(), tfidfAndIndex);
			}
			reader.close();

			SequenceFile.Reader reader1 = new SequenceFile.Reader(fs, dictPath,
					conf);
			Text key1 = new Text();
			LongWritable value1 = new LongWritable();
			while (reader1.next(key1, value1)) {
				dictWords.put(key1.toString(), value1.get());
			}
			reader1.close();

			super.setup(context);
		}

		/**
		 * 计算当前文档与所有文档中心之间的距离，选择最近的中心作为新的文档中心
		 * 
		 * @param context
		 *            输入 key:旧的文档中心 value:当前文档 输出 key:重新选择的文档中心 value:当前文档
		 */
		@Override
		protected void map(LongWritable key, Text value, Context context)
				throws IOException, InterruptedException {
			int nearestNum = 0;
			Map<Long, Double> doc = new HashMap<Long, Double>();
			String[] iat = value.toString().split("/");
			for (String string : iat) {
				doc.put(Long.parseLong(string.split(":")[0]),
						Double.parseDouble(string.split(":")[1]));
			}
			nearestNum = DocTool.returnNearestCentNum(doc, centers,
					dictWords.size());
			this.classCenter = new IntWritable(nearestNum);
			context.write(this.classCenter, value);
		}

	}

	public static class KmeansReducer extends
			Reducer<IntWritable, Text, IntWritable, Text> {

		public static enum Counter {
			CONVERGED
		}

		private static final DecimalFormat DF = new DecimalFormat(
				"###.########");
		private static Map<String, Long> dictWords = new HashMap<String, Long>();
		private Text cendroidTfidf = new Text();

		@Override
		protected void setup(Context context) throws IOException,
				InterruptedException {
			Configuration conf = context.getConfiguration();
			Path dictPath = new Path(conf.get("DICTPATH"));
			FileSystem fs = FileSystem.get(conf);
			SequenceFile.Reader reader = new SequenceFile.Reader(fs, dictPath,
					conf);
			Text key = new Text();
			LongWritable value = new LongWritable();
			while (reader.next(key, value)) {
				dictWords.put(key.toString(), value.get());
			}
			reader.close();
			super.setup(context);
		}

		/**
		 * 在得到一个文档集合后重新计算这个集合的中心
		 * 
		 * @param context
		 *            输入 key:文档中心 value:属于文档中心所在类的文档<文档1，文档2...文档n> 输出
		 *            key:新的文档中心 value:文档中心对应的文档
		 */
		@Override
		protected void reduce(IntWritable key, Iterable<Text> values,
				Context context) throws IOException, InterruptedException {

			double[] sumOfTfidf = new double[dictWords.size()];
			StringBuilder allTfidf = new StringBuilder();
			int belongNum = 0;
			for (int i = 0; i < sumOfTfidf.length; i++) {
				sumOfTfidf[i] = 0;
			}

			for (Text var : values) {
				String[] iat = var.toString().split("/");
				for (String string : iat) {
					sumOfTfidf[Integer.parseInt(string.split(":")[0])] += Double
							.parseDouble(string.split(":")[1]);
				}
				belongNum++;

			}

			for (int i = 0; i < sumOfTfidf.length; i++) {
				sumOfTfidf[i] = sumOfTfidf[i] / belongNum;
				if (sumOfTfidf[i] > 10E-6) {
					allTfidf.append(i + ":" + DF.format(sumOfTfidf[i]) + "/");
				}
			}

			this.cendroidTfidf.set(allTfidf.toString());

			context.write(key, this.cendroidTfidf);
		}
	}

	public static class LastKmeansMapper extends
			Mapper<LongWritable, Text, LongWritable, IntWritable> {

		private static Map<Integer, Map<Long, Double>> centers = new HashMap<Integer, Map<Long, Double>>();
		private static Map<String, Long> dictWords = new HashMap<String, Long>();
		private IntWritable classCenter;

		/**
		 * 将文档中心从hdfs中加载至内存
		 */
		@Override
		protected void setup(Context context) throws IOException,
				InterruptedException {
			Configuration conf = context.getConfiguration();
			Path centroids = new Path(conf.get("CENPATH"));
			Path dictPath = new Path(conf.get("DICTPATH"));
			FileSystem fs = FileSystem.get(conf);

			SequenceFile.Reader reader = new SequenceFile.Reader(fs, centroids,
					conf);
			IntWritable key = new IntWritable();
			Text value = new Text();
			while (reader.next(key, value)) {
				Map<Long, Double> tfidfAndIndex = new HashMap<Long, Double>();
				String[] iat = value.toString().split("/");
				for (String string : iat) {
					tfidfAndIndex.put(Long.parseLong(string.split(":")[0]),
							Double.parseDouble(string.split(":")[1]));
				}
				centers.put(key.get(), tfidfAndIndex);
			}
			reader.close();

			SequenceFile.Reader reader1 = new SequenceFile.Reader(fs, dictPath,
					conf);
			Text key1 = new Text();
			LongWritable value1 = new LongWritable();
			while (reader1.next(key1, value1)) {
				dictWords.put(key1.toString(), value1.get());
			}
			reader1.close();

			super.setup(context);
		}

		/**
		 * 计算当前文档与所有文档中心之间的距离，选择最近的中心作为新的文档中心
		 * 
		 * @param context
		 *            输入 key:旧的文档中心 value:当前文档 输出 key:重新选择的文档中心 value:当前文档
		 */
		@Override
		protected void map(LongWritable key, Text value, Context context)
				throws IOException, InterruptedException {
			int nearestNum = 0;
			Map<Long, Double> doc = new HashMap<Long, Double>();
			String[] iat = value.toString().split("/");
			for (String string : iat) {
				doc.put(Long.parseLong(string.split(":")[0]),
						Double.parseDouble(string.split(":")[1]));
			}
			nearestNum = DocTool.returnNearestCentNum(doc, centers,
					dictWords.size());
			this.classCenter = new IntWritable(nearestNum);
			context.write(key, this.classCenter);
		}

	}

	public static class LastKmeansReducer extends
			Reducer<LongWritable, IntWritable, LongWritable, IntWritable> {

		@Override
		protected void reduce(LongWritable key, Iterable<IntWritable> values,
				Context context) throws IOException, InterruptedException {
			super.reduce(key, values, context);
		}

	}

	public int run(String[] args) throws Exception {
		int iteration = 0;
		Configuration conf = new Configuration();
		conf.set("num.iteration", iteration + "");
		conf.set("DICTPATH", DICT_PATH);

		Path in = new Path(BASE_PATH + "/docvetor");
		Path center = new Path(CEN_PATH);
		conf.set("CENPATH", center.toString());
		Path out = new Path(BASE_PATH + "/clustering/depth_0");

		Job job = new Job(conf);
		job.setJobName("KMeansPrepare Clustering");

		job.setMapperClass(KmeansMapper.class);
		job.setReducerClass(KmeansReducer.class);
		job.setJarByClass(Kmeans.class);

		FileInputFormat.addInputPath(job, in);
		FileSystem fs = FileSystem.get(conf);
		if (fs.exists(out))
			fs.delete(out, true);

		if (fs.exists(center))
			fs.delete(out, true);

		if (fs.exists(in))
			fs.delete(out, true);

		FileOutputFormat.setOutputPath(job, out);
		job.setInputFormatClass(SequenceFileInputFormat.class);
		job.setOutputFormatClass(SequenceFileOutputFormat.class);

		job.setOutputKeyClass(IntWritable.class);
		job.setOutputValueClass(Text.class);

		job.waitForCompletion(true);

		// 是否需要继续迭代
		long counter = 5;
		// 迭代次数
		iteration++;
		while (counter > 0) {
			conf = new Configuration();
			conf.set("CENPATH", BASE_PATH + "/clustering/depth_"
					+ (iteration - 1) + "/" + "part-r-00000/");
			conf.set("num.iteration", iteration + "");
			conf.set("DICTPATH", DICT_PATH);
			job = new Job(conf);
			job.setJobName("KMeans Clustering " + iteration);

			job.setMapperClass(KmeansMapper.class);
			job.setReducerClass(KmeansReducer.class);
			job.setJarByClass(Kmeans.class);

			in = new Path(BASE_PATH + "/docvetor");
			out = new Path(BASE_PATH + "/clustering/depth_" + iteration);

			FileInputFormat.addInputPath(job, in);
			if (fs.exists(out))
				fs.delete(out, true);

			FileOutputFormat.setOutputPath(job, out);
			job.setInputFormatClass(SequenceFileInputFormat.class);
			job.setOutputFormatClass(SequenceFileOutputFormat.class);
			job.setOutputKeyClass(IntWritable.class);
			job.setOutputValueClass(Text.class);

			job.waitForCompletion(true);
			counter--;

			StringBuilder oldCentroid = new StringBuilder();
			StringBuilder newCentroid = new StringBuilder();
			Path oldCenPath = new Path(BASE_PATH + "/clustering/depth_"
					+ (iteration - 1) + "/part-r-00000");
			SequenceFile.Reader oldReader = new Reader(fs, oldCenPath, conf);
			IntWritable key = new IntWritable();
			Text value = new Text();
			while (oldReader.next(key, value)) {
				oldCentroid.append(key.toString() + value.toString());
			}
			oldReader.close();
			Path newCenPath = new Path(BASE_PATH + "/clustering/depth_"
					+ (iteration) + "/part-r-00000");
			SequenceFile.Reader newReader = new Reader(fs, newCenPath, conf);
			IntWritable key1 = new IntWritable();
			Text value1 = new Text();
			while (newReader.next(key1, value1)) {
				newCentroid.append(key1.toString() + value1.toString());
			}
			newReader.close();
			
			
			iteration++;
			
			if (newCentroid.toString().equals(oldCentroid.toString()))
				break;

		}

		conf = new Configuration();
		conf.set("CENPATH", BASE_PATH + "/clustering/depth_" + (iteration - 1)
				+ "/" + "part-r-00000/");
		conf.set("num.iteration", iteration + "");
		conf.set("DICTPATH", DICT_PATH);
		job = new Job(conf);
		job.setJobName("KMeans Last Clustering");

		job.setMapperClass(LastKmeansMapper.class);
		job.setReducerClass(LastKmeansReducer.class);
		job.setJarByClass(Kmeans.class);

		in = new Path(BASE_PATH + "/docvetor");
		out = new Path(BASE_PATH + "/result");

		FileInputFormat.addInputPath(job, in);
		if (fs.exists(out))
			fs.delete(out, true);

		FileOutputFormat.setOutputPath(job, out);
		job.setInputFormatClass(SequenceFileInputFormat.class);
		job.setOutputFormatClass(SequenceFileOutputFormat.class);
		job.setOutputKeyClass(LongWritable.class);
		job.setOutputValueClass(IntWritable.class);

		System.exit(job.waitForCompletion(true) ? 0 : 1);
		return 0;
	}

	public static void main(String[] args) throws Exception {
		int res = ToolRunner.run(new Configuration(), new Kmeans(), args);
		System.exit(res);
	}

}
