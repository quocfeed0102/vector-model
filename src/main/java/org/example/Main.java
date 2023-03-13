// N18DCC004 Hoang Nghia Quoc Anh
// N18DCCN163 Ho Mai Que
// N18DCCN166 Tran Anh Quoc

package org.example;

import edu.stanford.nlp.ling.CoreLabel;
import edu.stanford.nlp.pipeline.CoreDocument;
import edu.stanford.nlp.pipeline.StanfordCoreNLP;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.*;
import java.util.stream.Collectors;

public class Main {

    private static final Set<String> stopWords = new HashSet<String>(
            Arrays.asList(
                    "",
                    "a",
                    "is",
                    "the",
                    "of",
                    "all",
                    "to",
                    "can",
                    "be",
                    "as",
                    "once",
                    "for",
                    "at",
                    "am",
                    "are",
                    "has",
                    "have",
                    "had",
                    "up",
                    "his",
                    "her",
                    "in",
                    "on",
                    "no",
                    "we",
                    "do",
                    "by",
                    "or",
                    "and",
                    "not"
            )
    );

    private static final String fileDocumentPath = "doc.txt";

    private static final String fileQueryPath = "query.txt";

    private static final Map<String, Set<String>> documents = new HashMap<String, Set<String>>();

    private static final Map<String, Set<String>> queries = new HashMap<String, Set<String>>();

    private static final Map<String, List<Double>> vectorDocuments = new HashMap<String, List<Double>>();

    private static final Map<String, List<Double>> vectorQueries = new HashMap<String, List<Double>>();

    private static ArrayList<String> wordsList = new ArrayList<>();

    private static ArrayList<Integer> wordDocumentCounts = new ArrayList<>();

    public static void main(String[] args) {

        Scanner sc = new Scanner(System.in);
        readFile(documents, fileDocumentPath);
        readFile(queries, fileQueryPath);
        initVectorDocument();
        initWordDocumentCounts();

        String choice;
        do {
            System.out.println("Nhap exit de thoat!");
            System.out.print("Nhap lua chon: ");
            choice = sc.nextLine();
            switch (choice) {
                case "exit": {
                    break;
                }
                default: {
                    Map<String, Double> res = new HashMap<>();
                    System.out.println("choice: " + choice);
                    if (!vectorQueries.containsKey(choice)) {
                        initVectorQuery(choice);
                    }

                    double[] vectorQuery = vectorQueries.get(choice).stream().mapToDouble(Double::doubleValue).toArray();

                    for (Map.Entry<String, List<Double>> vectorDocument : vectorDocuments.entrySet()) {
                        double[] vectorDoc = vectorDocument.getValue().stream().mapToDouble(Double::doubleValue).toArray();
                        res.put(vectorDocument.getKey(), calculateCosineSimilarity(vectorDoc, vectorQuery));
                    }
//                    Collections.sort(res);

                    System.out.println(res);
                    List<Map.Entry<String, Double>> list = new ArrayList<>(res.entrySet());

                    Collections.sort(list, Map.Entry.comparingByValue());

                    Map<String, Double> sortedMap = new LinkedHashMap<>();

                    String indexDocument = "";
                    double value = 0;
                    for (Map.Entry<String, Double> entry : list) {
                        sortedMap.put(entry.getKey(), entry.getValue());
                        indexDocument = entry.getKey();
                        value = entry.getValue();
                    }
                    System.out.println("res sorted: " + sortedMap);
                    System.out.println("Ket qua: doc_index:" + indexDocument + " value:" + value);
                }
            }
        } while (!choice.equals("exit"));

    }

    public static void initVectorQuery(String index) {
        // Tính toán vector query sử dụng TF-IDF
        int[] queryVector = new int[wordsList.size()];
        String[] queryWords = queries.get(index).toArray(new String[0]);
        for (String queryWord : queryWords) {
            int tempIndex = wordsList.indexOf(queryWord);
            if (tempIndex != -1) {
                queryVector[tempIndex]++;
            }
        }

        // Tính toán TF-IDF cho vector query
        Double[] queryTFIDFVector = new Double[wordsList.size()];
        for (int i = 0; i < wordsList.size(); i++) {
            int tf = queryVector[i];
            double idf = Math.log((double) documents.size() / (double) wordDocumentCounts.get(i));
            queryTFIDFVector[i] = tf * idf;
        }
        vectorQueries.put(index, Arrays.asList(queryTFIDFVector));
        System.out.println("Query " + index + ": " + Arrays.toString(queryTFIDFVector));
    }

    public static void initWordDocumentCounts() {
        // Khởi tạo mảng documentCounts
        int[] documentCounts = new int[wordsList.size()];

// Đếm số lần xuất hiện của các từ trong tài liệu
        for (int i = 0; i < documents.size(); i++) {
            String[] words = documents.get(String.valueOf(i + 1)).toArray(new String[0]);
            ArrayList<String> uniqueWords = new ArrayList<>(Arrays.asList(words));
            uniqueWords = new ArrayList<>(uniqueWords.stream().distinct().collect(Collectors.toList()));
            for (String word : uniqueWords) {
                int index = wordsList.indexOf(word);
                if (index != -1) {
                    documentCounts[index]++;
                }
            }
        }

        // Hiển thị giá trị của wordDocumentCounts
        for (int i = 0; i < wordsList.size(); i++) {
            wordDocumentCounts.add(documentCounts[i]);
            System.out.println(wordsList.get(i) + " appears in " + wordDocumentCounts.get(i) + " documents");
        }

    }

    public static void initVectorDocument() {
        for (Map.Entry<String, Set<String>> document : documents.entrySet()) {
            String[] words = document.getValue().toArray(new String[0]);
            for (String word : words) {
                if (!wordsList.contains(word)) {
                    wordsList.add(word);
                }
            }
        }
        Collections.sort(wordsList);
        System.out.println("wordlist: " + wordsList);
        System.out.println("wordlist size: " + wordsList.size());
        double[][] tf = new double[documents.size()][wordsList.size()];
        for (int i = 0; i < documents.size(); i++) {
            String[] words = documents.get(String.valueOf(i + 1)).toArray(new String[0]);
            for (int j = 0; j < wordsList.size(); j++) {
                int count = 0;
                for (String word : words) {
                    if (word.equals(wordsList.get(j))) {
                        count++;
                    }
                }
                tf[i][j] = (double) count / words.length;
            }
        }
        double[] idf = new double[wordsList.size()];
        Arrays.fill(idf, 0);
        for (int j = 0; j < wordsList.size(); j++) {
            for (int i = 0; i < documents.size(); i++) {
                String[] words = documents.get(String.valueOf(i + 1)).toArray(new String[0]);
                for (String word : words) {
                    if (word.equals(wordsList.get(j))) {
                        idf[j]++;
                        break;
                    }
                }
            }
            idf[j] = Math.log(documents.size() / idf[j]) / Math.log(2);
        }

        double[][] tfidf = new double[documents.size()][wordsList.size()];
        for (int i = 0; i < documents.size(); i++) {
            for (int j = 0; j < wordsList.size(); j++) {
                tfidf[i][j] = tf[i][j] * idf[j];
            }
        }

        for (int i = 0; i < documents.size(); i++) {
            System.out.printf("Document %d: ", i + 1);
            vectorDocuments.put(String.valueOf(i + 1), new ArrayList<>());
            for (int j = 0; j < wordsList.size(); j++) {
                System.out.printf("%.4f ", tfidf[i][j]);
                vectorDocuments.get(String.valueOf(i + 1)).add(tfidf[i][j]);
            }
            System.out.println();
        }

    }

    public static void readFile(Map<String, Set<String>> map, String filePath) {
        try {
            FileReader fileReader = new FileReader(filePath);
            BufferedReader bufferedReader = new BufferedReader(fileReader);

            String line;
            String temp = "";
            String[] temps;
            Set<String> tempSet;
            boolean flag = true;
            int index = 0;
            while ((line = bufferedReader.readLine()) != null) {
                if (flag) {
                    index = Integer.parseInt(line.trim());
                    flag = false;
                } else {
                    if (line.trim().equals("/")) {
                        flag = true;
                        map.put("" + index, npl(temp));
                        temp = "";
                    } else {
                        temp = temp + " " + line.trim().toLowerCase();
                    }
                }
            }

            bufferedReader.close();
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    public static Set<String> npl(String text) {
        Properties props = new Properties();
        props.setProperty("annotators", "tokenize,ssplit,pos,lemma");
        StanfordCoreNLP pipeline = new StanfordCoreNLP(props);
        CoreDocument document = pipeline.processToCoreDocument(text);
        Set<String> result = new HashSet<String>();
        String temp;
        for (CoreLabel tok : document.tokens()) {
            temp = tok.lemma();
            if (!stopWords.contains(temp)) {
                result.add(temp.toLowerCase());
            }
        }
        return result;
    }

    public static double calculateCosineSimilarity(double[] v1, double[] v2) {
        double dotProduct = 0;
        double norm1 = 0;
        double norm2 = 0;

        for (int i = 0; i < v1.length; i++) {
            dotProduct += v1[i] * v2[i];
            norm1 += v1[i] * v1[i];
            norm2 += v2[i] * v2[i];
        }

        double cosineSimilarity = dotProduct / (Math.sqrt(norm1) * Math.sqrt(norm2));
        return cosineSimilarity;
    }
}
